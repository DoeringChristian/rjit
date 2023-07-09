use crate::Trace;

macro_rules! test_uop {
        ($jop:ident($init:expr; $ty:ident) $(,$mod:literal)?) => {
            test_uop!($ty::$jop => $jop($init; $ty) $(,$mod)?);
        };
        ($rop:expr => $jop:ident($init:expr; $ty:ident) $(,$mod:literal)?) => {
            paste::paste! {
                #[test]
                fn [<$jop _$ty $(__$mod)?>]() {

                    let initial: &[$ty] = &$init;

                    let ir = Trace::default();
                    ir.set_backend(["cuda"]);

                    let x = ir.[<buffer_$ty>](initial);

                    let y = x.$jop();

                    ir.schedule(&[&y]);

                    ir.eval();

                    insta::assert_snapshot!(ir.kernel_history());

                    for (i, calculated) in y.[<to_host_$ty>]().into_iter().enumerate(){
                        let expected = ($rop)(initial[i]);
                        approx::assert_abs_diff_eq!(calculated, expected, epsilon = 0.0001);
                    }
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
fn opaque() {
    let ir = Trace::default();
    ir.set_backend(["cuda"]);

    let x = ir.literal_u32(10);
    let x = x.opaque();

    let y = ir.buffer_u32(&[1, 2, 3]);

    let y = y.add(&x);

    y.schedule();
    ir.eval();

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host_u32(), vec![11, 12, 13]);
}
#[test]
fn make_opaque() {
    let ir = Trace::default();
    ir.set_backend(["cuda"]);

    let x = ir.literal_u32(10);
    x.make_opaque();

    let y = ir.buffer_u32(&[1, 2, 3]);

    let y = y.add(&x);

    y.schedule();
    ir.eval();

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host_u32(), vec![11, 12, 13]);
}

#[test]
fn compress_small() {
    let ir = Trace::default();
    ir.set_backend(["cuda"]);

    // let x = ir.array(&[true, false, true]);
    let x = ir.array(&[true, false, true, false, true, false, true, false]);

    let i = x.compress();

    assert_eq!(i.to_host_u32(), vec![0, 2, 4, 6]);
}
#[test]
fn compress_large() {
    let ir = Trace::default();
    ir.set_backend(["cuda"]);

    const N: u32 = 4100;

    let x = ir.array(&vec![true; N as usize]);

    let i = x.compress();

    assert_eq!(i.to_host_u32(), (0..N).collect::<Vec<_>>());
}
#[test]
fn compress_small_optix() {
    let ir = Trace::default();
    ir.set_backend(["optix"]);

    // let x = ir.array(&[true, false, true]);
    let x = ir.array(&[true, false, true, false, true, false, true, false]);

    let i = x.compress();

    assert_eq!(i.to_host_u32(), vec![0, 2, 4, 6]);
}
#[test]
fn compress_large_optix() {
    let ir = Trace::default();
    ir.set_backend(["optix"]);

    const N: u32 = 4100;

    let x = ir.array(&vec![true; N as usize]);

    let i = x.compress();

    assert_eq!(i.to_host_u32(), (0..N).collect::<Vec<_>>());
}
