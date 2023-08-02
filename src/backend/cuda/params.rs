use anyhow::{anyhow, Result};
use itertools::Itertools;
use std::ops::Range;

use crate::backend::optix;
use crate::schedule::Env;

use super::cuda;

pub fn params_cuda(size: usize, env: &Env) -> Result<Vec<u64>> {
    let sizes = [Ok(size as u64)].into_iter();
    let opaques = env.opaques().iter().map(|o| Ok(*o));
    let buffers = env.buffers().iter().map(|b| {
        b.buffer
            .downcast_ref::<cuda::Buffer>()
            .ok_or(anyhow!("Could not downcast Buffer!"))
            .map(|b| b.ptr())
    });
    let textures = env
        .textures()
        .iter()
        .map(|t| {
            t.downcast_ref::<cuda::Texture>()
                .ok_or(anyhow!("Could not downcast Texture!"))
                .map(|t| t.ptrs())
        })
        .flatten_ok();
    sizes
        .chain(opaques)
        .chain(buffers)
        .chain(textures)
        .collect()
}
pub fn params_optix(size: usize, env: &Env) -> Result<Vec<u64>> {
    let sizes = [Ok(bytemuck::cast::<_, u64>([size as u32, 0u32]))].into_iter();
    let opaques = env.opaques().iter().map(|o| Ok(*o));
    let buffers = env.buffers().iter().map(|b| {
        b.buffer
            .downcast_ref::<cuda::Buffer>()
            .ok_or(anyhow!("Could not downcast Buffer!"))
            .map(|b| b.ptr())
    });
    let textures = env
        .textures()
        .iter()
        .map(|t| {
            t.downcast_ref::<cuda::Texture>()
                .ok_or(anyhow!("Could not downcast Texture!"))
                .map(|t| t.ptrs())
        })
        .flatten_ok();
    let accels = env.accels().iter().map(|a| {
        a.downcast_ref::<optix::Accel>()
            .ok_or(anyhow!("Could not downcast Acceleration Structure!"))
            .map(|a| a.ptr())
    });
    sizes
        .chain(opaques)
        .chain(buffers)
        .chain(textures)
        .chain(accels)
        .collect()
}

pub struct ParamOffset {
    n_buffers: usize,
    n_opaques: usize,
    n_accels: usize,
    n_textures_internal: usize,
    texture_offsets: Vec<Range<usize>>,
}

impl ParamOffset {
    pub fn from_env(env: &Env) -> Self {
        let n_opaques = env.opaques().len();
        let n_buffers = env.buffers().len();
        let n_accels = env.accels().len();
        fn n_textures(tex: &dyn crate::backend::Texture) -> usize {
            (tex.n_channels() - 1) / 4 + 1
        }

        let texture_offsets = env
            .textures()
            .iter()
            .scan(0, |sum, tex| {
                let tmp = *sum;
                let n_textures = n_textures(tex.as_ref());
                *sum += n_textures;
                Some(tmp)
            })
            .zip(env.textures().iter().map(|tex| n_textures(tex.as_ref())))
            .map(|(start, length)| start..(start + length))
            .collect::<Vec<_>>();

        let n_textures_internal = texture_offsets.last().map(|l| l.end).unwrap_or(0);
        Self {
            n_buffers,
            n_opaques,
            n_accels,
            n_textures_internal,
            texture_offsets,
        }
    }
    pub fn opaque(&self, idx: usize) -> usize {
        assert!(idx < self.n_opaques);
        1 + idx
    }
    pub fn buffer(&self, idx: usize) -> usize {
        assert!(idx < self.n_buffers);
        1 + self.n_opaques + idx
    }
    pub fn texture_ranges(&self, idx: usize) -> Range<usize> {
        assert!(idx < self.texture_offsets.len());
        let texture_offset = 1 + self.n_opaques + self.n_buffers;
        let range = self.texture_offsets[idx].clone();
        (range.start + texture_offset)..(range.end + texture_offset)
    }
    pub fn accel(&self, idx: usize) -> usize {
        assert!(idx < self.n_accels);
        1 + self.n_opaques + self.n_buffers + self.n_textures_internal
    }
}
