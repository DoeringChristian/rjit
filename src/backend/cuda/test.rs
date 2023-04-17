use crate::jit::Jit;
use crate::trace::{ReduceOp, Trace};

#[test]
fn refcounting() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_f32(&[1.; 10]);
    assert_eq!(x.var().rc, 1, "rc of x should be 1 (in x)");
    let y = x.add(&x);
    // let y = ir::add(&x, &x);

    assert_eq!(
        x.var().rc,
        3,
        "rc of x should be 3 (2 refs in y and 1 in x)"
    );

    assert_eq!(y.var().rc, 1, "rc of y should be 1 (in y)");

    ir.schedule(&[&y]);
    let mut jit = Jit::default();

    assert_eq!(
        y.var().rc,
        2,
        "rc of y should be 2 (1 in y and 1 in schedule)"
    );

    jit.eval(&mut ir.borrow_mut());

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

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(y.to_host_f32(), vec![2f32; 10]);
}
#[test]
fn load_add_f32() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_f32(&[1.; 10]);
    // let y = ir::add(&x, &x);
    let y = x.add(&x);

    ir.schedule(&[&y]);
    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(y.to_host_f32(), vec![2f32; 10]);
}
#[test]
fn load_gather_f32() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_f32(&[1., 2., 3., 4., 5.]);
    let i = ir.buffer_u32(&[0, 1, 4]);
    let y = x.gather(&i, None);

    ir.schedule(&[&y]);
    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(y.to_host_f32(), vec![1., 2., 5.]);
}
#[test]
fn reindex() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.index(10);

    let i = ir.index(3);
    let c = ir.literal_u32(2);
    let i = i.add(&c);

    let y = x.gather(&i, None);

    ir.schedule(&[&y]);
    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(y.to_host_u32(), vec![2, 3, 4]);
}
#[test]
fn index() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let i = ir.index(10);

    ir.schedule(&[&i]);
    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(i.to_host_u32(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
#[test]
fn gather_eval() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let tmp_x;
    let tmp_y;
    let tmp_z;
    let r = {
        let x = ir.index(3);
        dbg!();
        assert_eq!(x.var().rc, 1);
        let y = ir.buffer_u32(&[1, 2, 3]);
        dbg!();
        assert_eq!(y.var().rc, 1);

        // let z = ir::add(&x, &y);
        let z = x.add(&y);
        dbg!();
        assert_eq!(x.var().rc, 2);
        assert_eq!(y.var().rc, 2);
        assert_eq!(z.var().rc, 1);

        let r = z.gather(&ir.index(3), None);
        dbg!();
        assert_eq!(x.var().rc, 2);
        assert_eq!(y.var().rc, 2);
        assert_eq!(z.var().rc, 3);
        assert_eq!(r.var().rc, 1);
        tmp_x = x.id();
        tmp_y = y.id();
        tmp_z = z.id();
        r
    };
    assert_eq!(r.var().rc, 1);
    assert_eq!(ir.borrow_mut().get_var(tmp_x).unwrap().rc, 1);
    assert_eq!(ir.borrow_mut().get_var(tmp_y).unwrap().rc, 1);
    assert_eq!(ir.borrow_mut().get_var(tmp_z).unwrap().rc, 2); // z is referenced by r and the
                                                               // schedule

    ir.schedule(&[&r]);
    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    assert_eq!(r.var().rc, 1);
    assert!(ir.borrow_mut().get_var(tmp_z).is_none());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(r.to_host_u32(), vec![1, 3, 5]);
}
#[test]
fn paralell() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.index(10);

    let y = ir.index(3);

    ir.schedule(&[&x, &y]);
    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(x.to_host_u32(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(y.to_host_u32(), vec![0, 1, 2]);
}
#[test]
fn load_gather() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_f32(&[1., 2., 3.]);

    ir.schedule(&[&x]);

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(x.to_host_f32(), vec![1., 2., 3.]);
}
#[test]
fn eval_scatter() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_u32(&[0, 0, 0, 0]);
    let c = ir.literal_u32(1);
    let x = x.add(&c); // x: [1, 1, 1, 1]

    let i = ir.index(3);
    let c = ir.literal_u32(1);
    let i = i.add(&c); // i: [1, 2, 3]

    let y = ir.literal_u32(2);

    y.scatter(&x, &i, None); // x: [1, 2, 2, 2]

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(x.to_host_u32(), vec![1, 2, 2, 2]);
}
#[test]
fn scatter_twice() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_u32(&[0, 0, 0, 0]);

    let i = ir.index(3);
    let c = ir.literal_u32(1);
    let i = i.add(&c);

    let y = ir.literal_u32(2);

    y.scatter(&x, &i, None);

    let i = ir.index(2);

    let y = ir.literal_u32(3);

    y.scatter(&x, &i, None);

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(x.to_host_u32(), vec![3, 3, 2, 2]);
}
#[test]
fn scatter_twice_add() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_u32(&[0, 0, 0, 0]);

    let i = ir.index(3);
    let c = ir.literal_u32(1);
    let i = i.add(&c);

    let y = ir.literal_u32(2);

    y.scatter(&x, &i, None);

    let i = ir.index(2);

    let y = ir.literal_u32(3);

    y.scatter(&x, &i, None);

    let c = ir.literal_u32(1);
    let x = x.add(&c);

    ir.schedule(&[&x]);

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(x.to_host_u32(), vec![4, 4, 3, 3]);
}
#[test]
fn scatter_reduce() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_u32(&[0, 0, 0, 0]);

    let i = ir.buffer_u32(&[0, 0, 0]);

    let y = ir.literal_u32(1);

    y.scatter_reduce(&x, &i, None, ReduceOp::Add);

    ir.schedule(&[&x]);

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(x.to_host_u32(), vec![3, 0, 0, 0]);
}
#[test]
fn tex_lookup() {
    let ir = Trace::default();
    ir.set_backend("cuda");

    let x = ir.buffer_f32(&[0.5]);
    let y = ir.buffer_f32(&[0.5]);

    let data = ir.buffer_f32(&[1.; 400]);

    let tex = data.to_texture(&[10, 10]);

    let res = tex.tex_lookup(&[&x, &y]);

    let r = res[0].clone();

    r.schedule();

    let mut jit = Jit::default();
    jit.eval(&mut ir.borrow_mut());

    insta::assert_snapshot!(jit.kernel_debug());

    assert_eq!(r.to_host_f32(), vec![1.]);
}
