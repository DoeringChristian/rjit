use std::mem::size_of;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::super::super::backend;
use super::cuda::Buffer;
use super::cuda_core::{Device, Module};

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
fn launch_config(device: &Device, size: u32) -> (u32, u32) {
    let max_threads = 128;
    let max_blocks_per_sm = 4u32;
    let num_sm = device.info().num_sm as u32;

    let blocks_avail = (size + max_threads - 1) / max_threads;

    let blocks;
    if blocks_avail < num_sm {
        // Not enough work for 1 full wave
        blocks = blocks_avail;
    } else {
        // Don't produce more than 4 blocks per SM
        let mut blocks_per_sm = blocks_avail / num_sm;
        if blocks_per_sm > max_blocks_per_sm {
            blocks_per_sm = max_blocks_per_sm;
        }
        blocks = blocks_per_sm * num_sm;
    }

    let mut threads = max_threads;
    if blocks <= 1 && size < max_threads {
        threads = size;
    }

    (blocks, threads)
}
pub fn compress(mask: &Buffer, kernels: &Module) -> Result<Arc<dyn backend::Buffer>> {
    let backend = mask.backend();
    let device = &backend.device;
    // let device = mask.device();

    let size = mask.size();
    let count_out = device
        .lease_buffer(size_of::<u32>())
        .ok_or(anyhow!("Could not create Buffer!"))?;
    // let count_out = Buffer::uninit(device, size_of::<u32>())?;
    let mut out = Buffer::uninit(backend, size as usize * size_of::<u32>())?;

    if size <= 4096 {
        let mut in_ptr = mask.ptr();
        let mut out_ptr = out.ptr();
        let mut count_out_ptr = count_out.ptr();
        let mut size = size as u32;

        let items_per_thread = 4;
        let thread_count = round_pow2((size + items_per_thread - 1) / items_per_thread);
        let shared_size = thread_count * 2 * std::mem::size_of::<u32>() as u32;
        let trailer = thread_count * items_per_thread - size;

        let mut params = [
            &mut in_ptr as *mut _ as *mut _,
            &mut out_ptr as *mut _ as *mut _,
            &mut size as *mut _ as *mut _,
            &mut count_out_ptr as *mut _ as *mut _,
        ];

        let stream = device.create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)?;

        unsafe {
            kernels.function("compress_small")?.launch(
                &stream,
                &mut params,
                1,
                thread_count,
                shared_size,
            )?;
        }
        stream.synchronize()?;
    } else {
        let stream = device.create_stream(cuda_rs::CUstream_flags::CU_STREAM_DEFAULT)?;

        let compress_large = kernels.function("compress_large")?;
        let scan_large_u32_int = kernels.function("scan_large_u32_init")?;

        let size = size as u32;
        let items_per_thread = 16u32;
        let thread_count = 128u32;
        let items_per_block = items_per_thread * thread_count;
        let block_count = (size + items_per_block - 1) / items_per_block;
        let shared_size = items_per_block * std::mem::size_of::<u32>() as u32;
        let mut scratch_items = block_count + 32;
        let trailer = items_per_block * block_count - size;

        let scratch = device
            .lease_buffer(scratch_items as usize * size_of::<u64>())
            .ok_or(anyhow!("Could not create buffer!"))?;

        let (block_count_init, thread_count_init) = launch_config(&device, scratch_items);

        let mut params = [
            &mut scratch.ptr() as *mut _ as *mut _,
            &mut scratch_items as *mut _ as *mut _,
        ];

        unsafe {
            scan_large_u32_int.launch(
                &stream,
                &mut params,
                block_count_init,
                thread_count_init,
                0,
            )?;
        }

        let mut in_ptr = mask.ptr();
        let mut out_ptr = out.ptr();
        let mut count_out_ptr = count_out.ptr();

        let mut scratch_ptr = scratch.ptr() + 32 * std::mem::size_of::<u64>() as u64;

        if trailer > 0 {
            unsafe {
                device
                    .ctx()
                    .cuMemsetD8Async(in_ptr + size as u64, 0, trailer as _, stream.raw())
                    .check()?;
            }
        }

        let mut params = [
            &mut in_ptr as *mut _ as *mut _,
            &mut out_ptr as *mut _ as *mut _,
            &mut scratch_ptr as *mut _ as *mut _,
            &mut count_out_ptr as *mut _ as *mut _,
        ];

        unsafe {
            compress_large.launch(&stream, &mut params, block_count, thread_count, shared_size)?;
        }

        stream.synchronize()?;
    }

    let mut size = 0u32;
    count_out.copy_to_host(bytemuck::cast_slice_mut(std::slice::from_mut(&mut size)));
    out.size = size as usize * size_of::<u32>();
    Ok(Arc::new(out))
}
