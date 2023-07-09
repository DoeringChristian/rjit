use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};

use crate::backend::{self, Backend};

type BackendFactory = Arc<dyn Fn() -> anyhow::Result<Box<dyn Backend>> + Send + Sync>;
// type BackendFactory = Arc<dyn Fn() -> i32 + Send + Sync>;

fn default_backends() -> HashMap<String, BackendFactory> {
    fn factory(
        name: &str,
        f: impl Fn() -> anyhow::Result<Box<dyn Backend>> + Send + Sync + 'static,
    ) -> (String, BackendFactory) {
        (String::from(name), Arc::new(f))
    }

    let reg: HashMap<String, BackendFactory> = HashMap::from([
        factory("cuda", || Ok(Box::new(backend::cuda::Backend::new()?))),
        factory("optix", || Ok(Box::new(backend::optix::Backend::new()?))),
    ]);

    reg
}

pub static BACKEND_REGISTRY: Lazy<Mutex<HashMap<String, BackendFactory>>> =
    Lazy::new(|| Mutex::new(default_backends()));

fn register_backend(
    name: &str,
    f: impl Fn() -> anyhow::Result<Box<dyn Backend>> + 'static + Send + Sync,
) -> anyhow::Result<()> {
    BACKEND_REGISTRY
        .lock()
        .insert(String::from(name), Arc::new(f));
    Ok(())
}
