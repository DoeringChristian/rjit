pub use super::*;

#[derive(Debug, Hash)]
pub struct ModuleInfo {
    pub asm: String,
    pub entry_point: String,
}
#[derive(Debug, Hash)]
pub struct HitGroupInfo {
    pub closest_hit: ModuleInfo,
    pub any_hit: Option<ModuleInfo>,
    pub intersection: Option<ModuleInfo>,
}
#[derive(Debug, Hash)]
pub struct MissGroupInfo {
    pub miss: ModuleInfo,
}
#[derive(Debug, Hash)]
pub struct SBTInfo {
    pub hit_groups: Vec<HitGroupInfo>,
    pub miss_groups: Vec<MissGroupInfo>,
}

impl<'a> From<SBTDesc<'a>> for SBTInfo {
    fn from(value: SBTDesc<'a>) -> Self {
        Self {
            hit_groups: value.hit_groups.iter().map(|g| (g).into()).collect(),
            miss_groups: value.miss_groups.iter().map(|g| (g).into()).collect(),
        }
    }
}

impl<'a> From<&'a ModuleDesc<'a>> for ModuleInfo {
    fn from(value: &'a ModuleDesc) -> Self {
        Self {
            asm: value.asm.into(),
            entry_point: value.entry_point.into(),
        }
    }
}

impl<'a> From<&'a HitGroupDesc<'a>> for HitGroupInfo {
    fn from(value: &'a HitGroupDesc<'a>) -> Self {
        Self {
            closest_hit: (&value.closest_hit).into(),
            any_hit: value.any_hit.as_ref().map(|m| m.into()),
            intersection: value.intersection.as_ref().map(|m| m.into()),
        }
    }
}
impl<'a> From<&'a MissGroupDesc<'a>> for MissGroupInfo {
    fn from(value: &'a MissGroupDesc<'a>) -> Self {
        Self {
            miss: (&value.miss).into(),
        }
    }
}
