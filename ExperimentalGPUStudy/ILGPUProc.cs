using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;

namespace ExperimentalGPUStudy
{
    class ILGPUProc
    {
        const int groupSize = 32;

        private static float[] FlatternArr(float[][] a)
        {
            float[] b = new float[a.Length * a.Length];
            int i = 0;
            foreach (var arr in a)
            {
                foreach (var el in arr)
                {
                    b[i++] = el;
                }
            }
            return b;
        }

        private static void MatrixMul(
            Index2 index, 
            ArrayView<float> a, 
            ArrayView<float> b, 
            ArrayView<float> c, int N)
        {
            int ix = index.X;
            int iy = index.Y;

            float sum = 0; 

            for (int r = 0; r < N; r++)
                sum += a[iy * N + r] * b[ix + r * N];

            c[iy * N + ix] = sum;
        }

        public static float[] RunMatrixMul(float[][] a, float[][] b, int N)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var matrixMulKernel = gpu.LoadAutoGroupedStreamKernel<
                Index2,
                ArrayView<float>,
                ArrayView<float>,
                ArrayView<float>,
                int>(MatrixMul);

            //Allocate memory
            int buffSize = N * N;
            MemoryBuffer<float> d_a = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_b = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_c = gpu.Allocate<float>(buffSize);

            d_a.CopyFrom(FlatternArr(a), 0, Index1.Zero, buffSize);
            d_b.CopyFrom(FlatternArr(b), 0, Index1.Zero, buffSize);

            matrixMulKernel(new Index2(N, N), d_a.View, d_b.View, d_c.View, N);

            // Wait for the kernel to finish...
            gpu.Synchronize();

            var c = d_c.GetAsArray();

            return c;
        }

        private static void MatrixMulShared(ArrayView<float> a, ArrayView<float> b, ArrayView<float> c, int N)
        {
            //int index = Grid.GlobalIndex.X;
            //if (index >= c.Length) return;

            int gx = Grid.GlobalIndex.X;
            int gy = Grid.GlobalIndex.Y;
            int lx = Group.IdxX;
            int ly = Group.IdxY;

            float sum = 0;

            var sa = SharedMemory.Allocate2D<float>(groupSize, groupSize);
            var sb = SharedMemory.Allocate2D<float>(groupSize, groupSize);

            for (int k = 0; k < N; k += groupSize)
            {
                sa[lx, ly] = a[gy * N + lx + k];
                sb[lx, ly] = b[(ly + k) * N + gx];
                Group.Barrier();
                for (int r = 0; r < groupSize; r++)
                {
                    sum += sa[r, ly] * sb[lx, r];
                }
                Group.Barrier();
            }
            c[gy * N + gx] = sum;
        }  

        public static float[] RunMatrixMulShared(float[][] a, float[][] b, int N)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var matrixMulKernelShared = gpu.LoadStreamKernel<
                ArrayView<float>,
                ArrayView<float>,
                ArrayView<float>,
                int>(MatrixMulShared);

            //Allocate memory
            var buffSize = N * N;
            MemoryBuffer<float> d_a = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_b = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_c = gpu.Allocate<float>(buffSize);

            d_a.CopyFrom(FlatternArr(a), 0, Index1.Zero, buffSize);
            d_b.CopyFrom(FlatternArr(b), 0, Index1.Zero, buffSize);

            //Groups per grid dimension
            int GrPerDim = (int)Math.Ceiling((float)N / groupSize);

            KernelConfig dimension = (
                                new Index2 (GrPerDim , GrPerDim), // Number of groups
                                new Index2(groupSize, groupSize)); // Group size (thread count in group)

            matrixMulKernelShared(dimension, d_a.View, d_b.View, d_c.View, N);

            // Wait for the kernel to finish...
            gpu.Synchronize();

            var c = d_c.GetAsArray();

            return c;

        }

        private static void FloydWarshall(Index1 index, int k, ArrayView<float> d_graphMinDist, int N)
        {
            int col = index % N;
            int row = index / N;

            float candidateBetterDist = d_graphMinDist[N * row + k] + d_graphMinDist[k * N + col];

            if (candidateBetterDist < d_graphMinDist[index]) 
                d_graphMinDist[index] = candidateBetterDist;            
        }

        public static float[] RunFloydWarshall(float[][] a, int N)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var floydWarshallKernel = gpu.LoadAutoGroupedStreamKernel<
                Index1,
                int,
                ArrayView<float>,
                int>(FloydWarshall);

            //Allocate memory
            var bufSize = N * N;
            MemoryBuffer<float> d_graphMinDist = gpu.Allocate<float>(bufSize);

            d_graphMinDist.CopyFrom(FlatternArr(a), 0, Index1.Zero, bufSize);

            for (int k = 0; k < N; k++)
            {
                floydWarshallKernel(bufSize, k, d_graphMinDist, N);
                gpu.Synchronize();
            }

            return d_graphMinDist.GetAsArray();
        }

        public static float[] RunMatrixMul(float[][] a, float[][] b, int N, ref Stopwatch sw)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var matrixMulKernel = gpu.LoadAutoGroupedStreamKernel<
                Index2,
                ArrayView<float>,
                ArrayView<float>,
                ArrayView<float>,
                int>(MatrixMul);

            //Allocate memory
            int buffSize = N * N;
            MemoryBuffer<float> d_a = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_b = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_c = gpu.Allocate<float>(buffSize);

            d_a.CopyFrom(FlatternArr(a), 0, Index1.Zero, buffSize);
            d_b.CopyFrom(FlatternArr(b), 0, Index1.Zero, buffSize);

            sw.Restart();

            matrixMulKernel(new Index2(N, N), d_a.View, d_b.View, d_c.View, N);

            // Wait for the kernel to finish...
            gpu.Synchronize();

            sw.Stop();

            var c = d_c.GetAsArray();

            return c;
        }

        public static float[] RunMatrixMulShared(float[][] a, float[][] b, int N, ref Stopwatch sw)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var matrixMulKernelShared = gpu.LoadStreamKernel<
                ArrayView<float>,
                ArrayView<float>,
                ArrayView<float>,
                int>(MatrixMulShared);

            //Allocate memory
            var buffSize = N * N;
            MemoryBuffer<float> d_a = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_b = gpu.Allocate<float>(buffSize);
            MemoryBuffer<float> d_c = gpu.Allocate<float>(buffSize);

            d_a.CopyFrom(FlatternArr(a), 0, Index1.Zero, buffSize);
            d_b.CopyFrom(FlatternArr(b), 0, Index1.Zero, buffSize);

            //Groups per grid dimension
            int GrPerDim = (int)Math.Ceiling((float)N / groupSize);

            KernelConfig dimension = (
                                new Index2(GrPerDim, GrPerDim), // Number of groups
                                new Index2(groupSize, groupSize)); // Group size (thread count in group)

            sw.Restart();

            matrixMulKernelShared(dimension, d_a.View, d_b.View, d_c.View, N);

            // Wait for the kernel to finish...
            gpu.Synchronize();

            sw.Stop();

            var c = d_c.GetAsArray();

            return c;

        }

        public static float[] RunFloydWarshall(float[][] a, int N, ref Stopwatch sw)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var floydWarshallKernel = gpu.LoadAutoGroupedStreamKernel<
                Index1,
                int,
                ArrayView<float>,
                int>(FloydWarshall);

            //Allocate memory
            var bufSize = N * N;
            MemoryBuffer<float> d_graphMinDist = gpu.Allocate<float>(bufSize);

            d_graphMinDist.CopyFrom(FlatternArr(a), 0, Index1.Zero, bufSize);

            sw.Restart();

            for (int k = 0; k < N; k++)
            {
                floydWarshallKernel(bufSize, k, d_graphMinDist, N);
                gpu.Synchronize();
            }

            sw.Stop();

            return d_graphMinDist.GetAsArray();
        }

        private static void OddEvenSort(Index1 index, ArrayView<float> a, VariableView<byte> stopFlag, int N, bool evenArr)
        {
            int i = index * 2;
            bool iterationEven = true;

            while (true)
            {
                if (stopFlag.Value > 0)
                    break;
                Group.Barrier();
                if (index == 0)
                    stopFlag.Value = 1;
                Group.Barrier();

                if (iterationEven)
                {
                    //swap
                    if (a[i] > a[i + 1])
                    {
                        var tmp = a[i];
                        a[i] = a[i + 1];
                        a[i + 1] = tmp;
                        stopFlag.Value = 0;
                    }
                }
                else
                {
                    if (!(evenArr && index == N / 2 - 1))
                    {
                        i++;
                        //swap
                        if (a[i] > a[i + 1])
                        {
                            var tmp = a[i];
                            a[i] = a[i + 1];
                            a[i + 1] = tmp;
                            stopFlag.Value = 0;
                        }
                        i--;
                    }
                }
                iterationEven = !iterationEven;
            }
        }

        public static float[] RunOddEvenSort(float[] a)
        {
            int N = a.Length;
            bool evenArr = (N % 2) == 0 ? true : false;

            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var oddEvenKernel = gpu.LoadAutoGroupedStreamKernel<
                Index1,
                ArrayView<float>,
                VariableView<byte>,
                int,
                bool>(OddEvenSort);

            //Allocate memory
            MemoryBuffer<float> d_a = gpu.Allocate<float>(N);
            MemoryBuffer<byte> d_stopFlag = gpu.AllocateZero<byte>(1);

            d_a.CopyFrom(a, 0, Index1.Zero, N);

            //Run kernel
            oddEvenKernel(N / 2, d_a, d_stopFlag.View.GetVariableView(0), N, evenArr);
            gpu.Synchronize();

            return d_a.GetAsArray();
        }

        private static void OddEvenSort2(Index1 index, ArrayView<float> a, VariableView<byte> stopFlag, bool iterationEven, int N, bool evenArr)
        {
            int i = index * 2;

            if (iterationEven)
            {
                //swap
                if (a[i] > a[i + 1])
                {
                    var tmp = a[i];
                    a[i] = a[i + 1];
                    a[i + 1] = tmp;
                    stopFlag.Value++;
                }
            }
            else
            {
                if (!(evenArr && index == N / 2 - 1))
                {
                    i++;
                    //swap
                    if (a[i] > a[i + 1])
                    {
                        var tmp = a[i];
                        a[i] = a[i + 1];
                        a[i + 1] = tmp;
                        stopFlag.Value++;
                    }
                    i--;
                }
            }
        }

        public static float[] RunOddEvenSort2(float[] a)
        {
            int N = a.Length;
            bool evenArr = (N % 2) == 0 ? true : false;

            bool stopFlag = false;
            bool iterationEven = true;

            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var oddEvenKernel = gpu.LoadAutoGroupedStreamKernel<
                Index1,
                ArrayView<float>,
                VariableView<byte>,
                bool,
                int,
                bool>(OddEvenSort2);

            //Allocate memory
            MemoryBuffer<float> d_a = gpu.Allocate<float>(N);
            MemoryBuffer<byte> d_stopFlag = gpu.AllocateZero<byte>(1);

            d_a.CopyFrom(a, 0, Index1.Zero, N);

            //Run kernel
            byte[] zero_val = new byte[1];
            zero_val[0] = 0;

            while (true)
            {
                if (stopFlag)
                    break;
                stopFlag = true;
                
                d_stopFlag.CopyFrom(zero_val, 0, 0, 1);
                oddEvenKernel(N / 2, d_a, d_stopFlag.View.GetVariableView(), iterationEven, N, evenArr);
                if(d_stopFlag.GetAsArray()[0] > 0)
                    stopFlag = false;

                iterationEven = !iterationEven;
            }

            return d_a.GetAsArray();
        }
    }
}
