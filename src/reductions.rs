use std::any::Any;
use std::sync::Arc;

use crate::backend::{cuda, DeviceFuture};
use crate::{VarRef, VarType};

fn round_pow2(mut x: u32) -> u32 {
    x = x.wrapping_sub(1);
    x |= x.wrapping_shr(1);
    x |= x.wrapping_shr(2);
    x |= x.wrapping_shr(4);
    x |= x.wrapping_shr(8);
    x |= x.wrapping_shr(16);
    x = x.wrapping_add(1);
    x
}

impl VarRef {
    // pub fn compress(&self) -> Self {
    //     assert_eq!(self.ty(), VarType::Bool);
    //     match self.backend_ident() {
    //         "CUDA" | "OptiX" => {
    //             let size = self.size();
    //             if size <= 4096 {
    //                 let compress_small = self
    //                     .ir
    //                     .register_kernel(
    //                         std::include_str!("backend/cuda/kernels/kernels_70.ptx"),
    //                         "compress_small",
    //                     )
    //                     .downcast_arc::<cuda::Kernel>()
    //                     .unwrap();
    //
    //                 self.schedule();
    //                 self.ir.eval();
    //
    //                 let out = self.ir.array_uninit::<u32>(self.size());
    //                 let count_out = self.ir.array_uninit::<u32>(1);
    //                 let mut in_ptr = self.ptr().unwrap();
    //                 let mut out_ptr = out.ptr().unwrap();
    //                 let mut count_out_ptr = count_out.ptr().unwrap();
    //                 let mut size = self.size() as u32;
    //
    //                 let items_per_thread = 4;
    //                 let thread_count = round_pow2((size + items_per_thread - 1) / items_per_thread);
    //                 let shared_size = thread_count * 2 * std::mem::size_of::<u32>() as u32;
    //                 let trailer = thread_count * items_per_thread - size;
    //
    //                 let mut params = [
    //                     &mut in_ptr as *mut _ as *mut _,
    //                     &mut out_ptr as *mut _ as *mut _,
    //                     &mut size as *mut _ as *mut _,
    //                     &mut count_out_ptr as *mut _ as *mut _,
    //                 ];
    //                 compress_small
    //                     .launch(&mut params, 1, thread_count, shared_size)
    //                     .wait();
    //
    //                 let size = count_out.to_host_u32()[0];
    //                 out.var().size = size as _;
    //
    //                 out
    //             } else {
    //                 let compress_large = self
    //                     .ir
    //                     .register_kernel(
    //                         std::include_str!("backend/cuda/kernels/kernels_70.ptx"),
    //                         "compress_large",
    //                     )
    //                     .downcast_arc::<cuda::Kernel>()
    //                     .unwrap();
    //                 let scan_large_u32_init = self
    //                     .ir
    //                     .register_kernel(
    //                         std::include_str!("backend/cuda/kernels/kernels_70.ptx"),
    //                         "scan_large_u32_init",
    //                     )
    //                     .downcast_arc::<cuda::Kernel>()
    //                     .unwrap();
    //
    //                 let size = size as u32;
    //                 let items_per_thread = 16u32;
    //                 let thread_count = 128u32;
    //                 let items_per_block = items_per_thread * thread_count;
    //                 let block_count = (size + items_per_block - 1) / items_per_block;
    //                 let shared_size = items_per_block * std::mem::size_of::<u32>() as u32;
    //                 let mut scratch_items = block_count + 32;
    //                 let trailer = items_per_block * block_count - size;
    //
    //                 dbg!(scratch_items * 4);
    //                 dbg!(block_count);
    //                 dbg!(shared_size);
    //
    //                 let scratch = self
    //                     .ir
    //                     .backend()
    //                     .buffer_uninit(scratch_items as usize * std::mem::size_of::<u64>());
    //                 let scratch = scratch.downcast_ref::<cuda::Buffer>().unwrap();
    //
    //                 let (block_count_init, thread_count_init) =
    //                     compress_large.launch_size(scratch_items as usize);
    //
    //                 let mut params = [
    //                     &mut scratch.ptr() as *mut _ as *mut _,
    //                     &mut scratch_items as *mut _ as *mut _,
    //                 ];
    //
    //                 dbg!(scratch.ptr());
    //
    //                 scan_large_u32_init
    //                     .launch(&mut params, block_count_init, thread_count_init, 0)
    //                     .wait();
    //
    //                 let out = self.ir.array_uninit::<u32>(size as _);
    //                 let count_out = self.ir.array(&[0u32]);
    //                 let mut in_ptr = self.ptr().unwrap();
    //                 let mut out_ptr = out.ptr().unwrap();
    //                 let mut count_out_ptr = count_out.ptr().unwrap();
    //
    //                 let mut scratch_ptr = scratch.ptr() + 32 * std::mem::size_of::<u64>() as u64;
    //
    //                 let mut params = [
    //                     &mut in_ptr as *mut _ as *mut _,
    //                     &mut out_ptr as *mut _ as *mut _,
    //                     &mut scratch_ptr as *mut _ as *mut _,
    //                     &mut count_out_ptr as *mut _ as *mut _,
    //                 ];
    //
    //                 dbg!(in_ptr);
    //                 dbg!(out_ptr);
    //                 dbg!(scratch_ptr);
    //                 dbg!(count_out_ptr);
    //                 dbg!(block_count);
    //                 dbg!(thread_count);
    //
    //                 std::fs::write("/tmp/dbg.txt", std::format!("in_ptr = {in_ptr:#018x}, out_ptr = {out_ptr:#018x}, scratch_ptr = {scratch_ptr:#018x}, count_out_ptr = {count_out_ptr:#018x}")).unwrap();
    //
    //                 let f =
    //                     compress_large.launch(&mut params, block_count, thread_count, shared_size);
    //
    //                 f.wait();
    //
    //                 drop(scratch);
    //                 // .wait();
    //
    //                 let size = count_out.to_host_u32()[0];
    //                 out.var().size = size as _;
    //
    //                 out
    //             }
    //         }
    //         _ => todo!(),
    //     }
    // }
}

#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn compress_small() {
        let ir = Trace::default();
        ir.set_backend("cuda");

        // let x = ir.array(&[true, false, true]);
        let x = ir.array(&[true, false, true, false, true, false, true, false]);

        let i = x.compress();

        assert_eq!(i.to_host_u32(), vec![0, 2, 4, 6]);
    }
    // #[test]
    // fn compress_large() {
    //     let ir = Trace::default();
    //     ir.set_backend("cuda");
    //
    //     // let x = ir.array(&[true, false, true]);
    //     // let x = ir.array(&vec![true; 4096 * 4]);
    //     // let x = ir.array(&vec![true; 4096 + 1024 + 512]);
    //     let x = ir.array(&vec![true; 4096 + 1]);
    //
    //     let i = x.compress();
    //
    //     i.schedule();
    //     ir.eval();
    //
    //     dbg!(i.size());
    //     // dbg!(i.to_host_u32());
    //
    //     // println!("{:?}", i.to_host_u32());
    //     // assert_eq!(i.to_host_u32(), (0..4098).collect::<Vec<_>>());
    // }
}
