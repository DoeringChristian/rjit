use cuda_builder::CudaBuilder;
fn main() {
    CudaBuilder::new("../../cuda-kernels")
        .copy_to("../../")
        .build()
        .unwrap();
}
