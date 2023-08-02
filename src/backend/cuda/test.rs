use crate::jit::Jit;
use crate::trace::{ReduceOp, Trace};
use anyhow::Result;

#[test]
fn refcounting() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[1f32; 10])?;
    assert_eq!(x.var().rc, 1, "rc of x should be 1 (in x)");
    let y = x.add(&x)?;
    // let y = ir::add(&x, &x);

    assert_eq!(
        x.var().rc,
        3,
        "rc of x should be 3 (2 refs in y and 1 in x)"
    );

    assert_eq!(y.var().rc, 1, "rc of y should be 1 (in y)");

    ir.schedule(&[&y]);

    assert_eq!(
        y.var().rc,
        2,
        "rc of y should be 2 (1 in y and 1 in schedule)"
    );

    ir.eval()?;

    assert_eq!(
        x.var().rc,
        1,
        "rc of x should be 1 (dependencies of y shuld be cleaned)"
    );
    assert_eq!(
        y.var().rc,
        1,
        "rc of y should be 2 (y from schedule should be cleaned)"
    );

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<f32>().unwrap(), vec![2f32; 10]);
    Ok(())
}
#[test]
fn load_add_f32() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[1f32; 10])?;
    // let y = ir::add(&x, &x);
    let y = x.add(&x)?;

    ir.schedule(&[&y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<f32>().unwrap(), vec![2f32; 10]);
    Ok(())
}
#[test]
fn load_gather_f32() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[1f32, 2., 3., 4., 5.])?;
    let i = ir.array(&[0u32, 1, 4])?;
    let y = x.gather(&i, None)?;

    ir.schedule(&[&y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<f32>().unwrap(), vec![1., 2., 5.]);
    Ok(())
}
#[test]
fn reindex() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.index(10);

    let i = ir.index(3);
    let c = ir.literal(2u32)?;
    let i = i.add(&c)?;

    let y = x.gather(&i, None)?;

    ir.schedule(&[&y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(y.to_host::<u32>().unwrap(), vec![2, 3, 4]);
    Ok(())
}
#[test]
fn index() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(&["cuda"])?;

    let i = ir.index(10);

    ir.schedule(&[&i]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(
        i.to_host::<u32>().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}
#[test]
fn gather_eval() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let r = {
        let x = ir.index(3);
        let y = ir.array(&[1u32, 2, 3])?;

        // let z = ir::add(&x, &y);
        let z = x.add(&y)?;

        let r = z.gather(&ir.index(3), None)?;
        r
    };
    // schedule

    ir.schedule(&[&r]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(r.to_host::<u32>().unwrap(), vec![1, 3, 5]);
    Ok(())
}
#[test]
fn paralell() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.index(10);

    let y = ir.index(3);

    ir.schedule(&[&x, &y]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(
        x.to_host::<u32>().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(y.to_host::<u32>().unwrap(), vec![0, 1, 2]);
    Ok(())
}
#[test]
fn _load_gather() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[1f32, 2., 3.])?;

    ir.schedule(&[&x]);
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<f32>().unwrap(), vec![1., 2., 3.]);
    Ok(())
}
#[test]
fn eval_scatter() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;
    let c = ir.literal(1u32)?;
    let x = x.add(&c)?; // x: [1, 1, 1, 1]

    let i = ir.index(3);
    let c = ir.literal(1u32)?;
    let i = i.add(&c)?; // i: [1, 2, 3]

    let y = ir.literal(2u32)?;

    y.scatter(&x, &i, None)?; // x: [1, 2, 2, 2]

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![1, 2, 2, 2]);
    Ok(())
}
#[test]
fn scatter_twice() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;

    let i = ir.index(3);
    let c = ir.literal(1u32)?;
    let i = i.add(&c)?;

    let y = ir.literal(2u32)?;

    y.scatter(&x, &i, None)?;

    let i = ir.index(2);

    let y = ir.literal(3u32)?;

    y.scatter(&x, &i, None)?;

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![3, 3, 2, 2]);
    Ok(())
}
#[test]
fn scatter_twice_add() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;

    let i = ir.index(3);
    let c = ir.literal(1u32)?;
    let i = i.add(&c)?;

    let y = ir.literal(2u32)?;

    y.scatter(&x, &i, None)?;

    let i = ir.index(2);

    let y = ir.literal(3u32)?;

    y.scatter(&x, &i, None)?;

    let c = ir.literal(1u32)?;
    let x = x.add(&c)?;

    ir.schedule(&[&x]);

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![4, 4, 3, 3]);
    Ok(())
}
#[test]
fn scatter_reduce() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[0u32, 0, 0, 0])?;

    let i = ir.array(&[0u32, 0, 0])?;

    let y = ir.literal(1u32)?;

    y.scatter_reduce(&x, &i, None, ReduceOp::Add)?;

    ir.schedule(&[&x]);

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u32>().unwrap(), vec![3, 0, 0, 0]);
    Ok(())
}
#[test]
fn tex_lookup() -> Result<()> {
    pretty_env_logger::try_init().ok();
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[0.5f32])?;
    let y = ir.array(&[0.5f32])?;

    let data = ir.array(&[1.0f32; 400])?;

    let tex = data.to_texture(&[10, 10], 4)?;

    let res = tex.tex_lookup(&[&x, &y])?;

    let r = res[0].clone();

    r.schedule();

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(r.to_host::<f32>().unwrap(), vec![1.]);
    Ok(())
}
#[test]
fn cast() -> Result<()> {
    pretty_env_logger::try_init().ok();
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[1u32, 2, 3])?;
    let x = x.cast(&crate::VarType::U64)?;

    x.schedule();

    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<u64>().unwrap(), vec![1, 2, 3]);
    Ok(())
}
#[test]
fn bitcast() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.array(&[0x3f800000u32, 0x40000000, 0x40400000])?;

    let x = x.bitcast(&crate::VarType::F32)?;

    x.schedule();
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(x.to_host::<f32>().unwrap(), vec![1., 2., 3.]);
    Ok(())
}
#[test]
fn scatter_const_mask() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let x = ir.literal(1f32)?;
    let mask = ir.array(&[true, false, true])?;

    let dst = ir.array(&[0f32, 0., 0.])?;

    x.scatter(&dst, &ir.index(3), Some(&mask))?;

    ir.eval()?;
    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(dst.to_host::<f32>().unwrap(), vec![1., 0., 1.]);
    Ok(())
}
#[test]
fn scatter_gather() -> Result<()> {
    let ir = Trace::default();
    ir.set_backend(["cuda"])?;

    let dst = ir.array(&[0u32, 0, 0])?;
    let idx = ir.index(3);

    let x = ir.array(&[1u32, 1, 1])?;

    x.scatter(&dst, &idx, None)?;

    let y = dst.gather(&idx, None)?;

    y.schedule();
    ir.eval()?;

    insta::assert_snapshot!(ir.kernel_history());

    assert_eq!(dst.to_host::<u32>().unwrap(), vec![1, 1, 1]);
    assert_eq!(y.to_host::<u32>().unwrap(), vec![1, 1, 1]);
    Ok(())
}
