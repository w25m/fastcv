#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <nvtx3/nvToolsExt.h>

struct Gauss_gen{
    thrust::pair<int,float> gauss_pair; //1-center,2-sigma

    __device__ float operator()(int i){
        int x=i-gauss_pair.first;
        return exp(-(x*x)/(2.0f*gauss_pair.second*gauss_pair.second));
    }
};

struct Gauss_gen_norm{
    thrust::pair<int,float> gauss_pair; //1-center,2-sigma
    float sum;

    __device__ float operator()(int i){
        int x=i-gauss_pair.first;
        float gauss=exp(-(x*x)/(2.0f*gauss_pair.second*gauss_pair.second));
        return gauss/sum;
    }
};

__global__ void Horizontal(uchar3* in, uchar3* out, float* kernel, int width, int height, int ksize) {
    extern __shared__ uchar3 tile[];
    int half = ksize/2;
    float sumR= 0, sumG=0, sumB=0;

    int tile_w= blockDim.x + 2* half;
    int idx= threadIdx.y *tile_w + threadIdx.x+half;

    int x= blockIdx.x* blockDim.x+ threadIdx.x;
    int y= blockIdx.y* blockDim.y+ threadIdx.y;

    if (x<width && y<height) //zabezpieczenie
        tile[idx]= in[y*width+x];
    else
        tile[idx]= make_uchar3(0,0,0);

    if(threadIdx.x<half){ //on the left from middle pixel
        int px= x-half;
        if (px<0) px=0;
        tile[threadIdx.y*tile_w +threadIdx.x]= in[y*width+px];
    }

    if(threadIdx.x>=blockDim.x - half){ // on the right from middle pixel
        int px= x+half;
        if(px>=width) px=width-1;
        tile[threadIdx.y*tile_w +threadIdx.x + 2*half] = in[y*width+px];
    }
    __syncthreads();
    if (x<width && y<height) {
        for(int i=0; i<ksize;i++){
            uchar3 pixel= tile[idx +(i-half)];
            sumR= sumR+pixel.z*kernel[i];
            sumG= sumG+ pixel.y*kernel[i];
            sumB = sumB+pixel.x*kernel[i];
        }
        out[y*width+x] = make_uchar3(sumB,sumG,sumR);
    }
}

__global__ void Vertical(uchar3* in, uchar3* out, float* kernel, int width, int height, int ksize) {
    extern __shared__ uchar3 tile[];
    int half = ksize/2;
    float sumR= 0, sumG=0, sumB=0;

    int tile_h= blockDim.y + 2*half;
    int idx= (threadIdx.y+ half)*blockDim.x +threadIdx.x;

    int x= blockIdx.x* blockDim.x+ threadIdx.x;
    int y= blockIdx.y* blockDim.y+ threadIdx.y;

    if (x<width && y<height){ //for safety
        tile[idx]= in[y*width+x];}
    else {
        tile[idx] = make_uchar3(0,0,0);}

    if (threadIdx.y<half) {// higher than middle
        int py= y-half;
        if (py<0) py=0;
        tile[threadIdx.y* blockDim.x+ threadIdx.x]= in[py* width+ x];
    }

    if (threadIdx.y >=blockDim.y -half){ //lower than middle
        int py= y+half;
        if(py>=height) py=height-1;
        tile[(threadIdx.y + 2*half)*blockDim.x + threadIdx.x]= in[py*width+x];
    }

    __syncthreads();

    if (x<width && y<height) {
        for(int i=0; i<ksize;i++){
            uchar3 pixel= tile[idx +(i-half)*blockDim.x];
            sumR= sumR+pixel.z*kernel[i];
            sumG= sumG+ pixel.y*kernel[i];
            sumB = sumB+pixel.x*kernel[i];
        }
        out[y*width+x] = make_uchar3(sumB,sumG,sumR);
    }
}


torch::Tensor gaussian_blur(torch::Tensor input, int k_width, int k_height, float sigma){
    int height = input.size(0);
    int width = input.size(1);
    
    int center_h = k_width/2;
    int center_v = k_height/2;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    thrust::device_vector<float> d_kernel_h(k_width);

    thrust::counting_iterator<int> count(0);
    auto gauss_vec_h=thrust::make_transform_iterator(count,Gauss_gen{thrust::make_pair(center_h,sigma)});//generates vector with gauss values
    float sum_h=thrust::reduce(thrust::device,gauss_vec_h,gauss_vec_h+k_width);//sum of vector values

    thrust::transform(thrust::device,count,count+k_width,d_kernel_h.begin(),Gauss_gen_norm{thrust::make_pair(center_h,sigma),sum_h});
    float* d_kernel_horizontal=thrust::raw_pointer_cast(d_kernel_h.data());

    thrust::device_vector<float> d_kernel_v(k_height);

    auto gauss_vec_v=thrust::make_transform_iterator(count,Gauss_gen{thrust::make_pair(center_v,sigma)});
    float sum_v=thrust::reduce(thrust::device,gauss_vec_v,gauss_vec_v+k_height);

    thrust::transform(thrust::device,count,count+k_height,d_kernel_v.begin(),Gauss_gen_norm{thrust::make_pair(center_v,sigma),sum_v});
    float* d_kernel_vertical=thrust::raw_pointer_cast(d_kernel_v.data());
    
    uchar3 *d_input, *d_temp, *d_output;
    size_t imgSize= width* height* sizeof(uchar3);

    cudaError_t err = cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_temp, imgSize);
    cudaMalloc(&d_output, imgSize);

    cudaMemcpyAsync(d_input, input.data_ptr(), imgSize, cudaMemcpyHostToDevice, stream);
    
    dim3 block(16,16);
    dim3 grid((width+ block.x-1)/ block.x, (height+ block.y-1)/ block.y);

    int half_hor= k_width/2;
    int tile_hor=block.x+2*half_hor;
    size_t sharedHor= tile_hor*block.y*sizeof(uchar3);

    int half_ver = k_height/2;
    int tile_ver= block.y+2*half_ver;
    size_t sharedVer= tile_ver*block.x*sizeof(uchar3);

    Horizontal<<<grid, block, sharedHor, stream>>>(d_input, d_temp, d_kernel_horizontal, width, height, k_width);    

    Vertical<<<grid, block, sharedVer, stream>>>(d_temp, d_output, d_kernel_vertical, width, height, k_height);

    cudaStreamSynchronize(stream);

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor output = torch::empty({height, width, 3}, options);
    
    cudaMemcpyAsync(output.data_ptr(), d_output, imgSize, cudaMemcpyDeviceToHost, stream);


    nvtxRangePush("Synchronizacja_Strumieni");
    cudaStreamSynchronize(stream);
   

    nvtxRangePush("Czyszczenie");
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);


    return output;
}
