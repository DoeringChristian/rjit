use crate::Trace;
use anyhow::*;

macro_rules! test_uop {
        ($jop:ident($init:expr; $ty:ident) $(,$mod:literal)?) => {
            test_uop!($ty::$jop => $jop($init; $ty) $(,$mod)?);
        };
        ($rop:expr => $jop:ident($init:expr; $ty:ident) $(,$mod:literal)?) => {
            paste::paste! {
                #[test]
                fn [<$jop _$ty $(__$mod)?>]() -> Result<()> {
                    // pretty_env_logger::init();

                    let initial: &[$ty] = &$init;

                    let ir = Trace::default();
                    ir.set_backend(["cuda"])?;

                    let x = ir.array(initial)?;

                    let y = x.$jop()?;

                    ir.schedule(&[&y]);

                    ir.eval()?;

                    insta::assert_snapshot!(ir.kernel_history());

                    for (i, calculated) in y.to_host::<$ty>()?.into_iter().enumerate(){
                        let expected = ($rop)(initial[i]);
                        approx::assert_abs_diff_eq!(calculated, expected, epsilon = 0.0001);
                    }
                    Ok(())
                }
            }
        };
    }

test_uop!(|x:f32| {1./x} => rcp(         [0.1, 0.5, 1., std::f32::consts::PI]; f32));
test_uop!(|x:f32| {1./x.sqrt()} => rsqrt([0.1, 0.5, 1., std::f32::consts::PI]; f32));
test_uop!(sin(                           [0.1, 0.5, 1., std::f32::consts::PI]; f32));
test_uop!(cos(                           [0.1, 0.5, 1., std::f32::consts::PI]; f32));
test_uop!(exp2(                          [0.1, 0.5, 1., std::f32::consts::PI]; f32));
test_uop!(log2(                          [0.1, 0.5, 1., std::f32::consts::PI]; f32));

#[test]
fn opaque() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.literal(10u32)?;
    let x = x.opaque();

    let y = ir.array(&[1u32, 2, 3])?;

    let y = y.add(&x)?;

    y.schedule();
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<u32>()?, vec![11u32, 12, 13]);
    Ok(())
}
#[test]
fn make_opaque() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.literal(10u32)?;
    x.make_opaque();

    let y = ir.array(&[1u32, 2, 3])?;

    let y = y.add(&x)?;

    y.schedule();
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<u32>().unwrap(), vec![11, 12, 13]);
    Ok(())
}

#[test]
fn compress_small() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    // let x = ir.array(&[true, false, true]);
    let x = ir.array(&[true, false, true, false, true, false, true, false])?;

    let i = x.compress()?;

    assert_eq!(i.to_host::<u32>().unwrap(), vec![0, 2, 4, 6]);
    Ok(())
}
#[test]
fn compress_large() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    const N: u32 = 4100;

    let x = ir.array(&vec![true; N as usize])?;

    let i = x.compress()?;

    assert_eq!(i.to_host::<u32>().unwrap(), (0..N).collect::<Vec<_>>());
    Ok(())
}
#[test]
fn compress_small_optix() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    // let x = ir.array(&[true, false, true]);
    let x = ir.array(&[true, false, true, false, true, false, true, false])?;

    let i = x.compress()?;

    assert_eq!(i.to_host::<u32>().unwrap(), vec![0, 2, 4, 6]);
    Ok(())
}
#[test]
fn compress_large_optix() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    const N: u32 = 4100;

    let x = ir.array(&vec![true; N as usize])?;

    let i = x.compress()?;

    assert_eq!(i.to_host::<u32>().unwrap(), (0..N).collect::<Vec<_>>());
    Ok(())
}
#[test]
fn kernel_reuse() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["optix"])?;

    for _ in 0..100 {
        let mut x = ir.array(&[0, 1, 2, 3, 4])?;
        for j in 0..100 {
            x = x.add(&ir.literal(j)?)?;
        }

        x.schedule();
        ir.eval()?;
    }

    println!("{}", ir.kernel_history());
    assert_eq!(ir.kernel_cache_size(), 1);

    Ok(())
}
