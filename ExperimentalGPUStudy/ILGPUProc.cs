using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ExperimentalGPUStudy
{
    class ILGPUProc
    {
        const int groupSize = 32;

        private static float[] FlatternArr(float[,] a)
        {
            float[] b = new float[a.Length];
            int i = 0;
            foreach (var el in a)
            {
                b[i++] = el;
            }
            return b;
        }

        private static void MatrixMul(Index2 index, ArrayView2D<float> a, ArrayView2D<float> b, ArrayView2D<float> c, int N)
        {
            int ix = index.X;
            int iy = index.Y;

            float sum = 0;

            for (int r = 0; r < N; r++)
            {
                sum += a[r, iy] * b[ix, r];
            }

            c[ix, iy] = sum;
        }

        public static float[] RunMatrixMul(float[,] a, float[,] b, int N)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var matrixMulKernel = gpu.LoadAutoGroupedStreamKernel<
                Index2,
                ArrayView2D<float>,
                ArrayView2D<float>,
                ArrayView2D<float>,
                int>(MatrixMul);

            //Allocate memory
            MemoryBuffer2D<float> d_a = gpu.Allocate<float>(N, N);
            MemoryBuffer2D<float> d_b = gpu.Allocate<float>(N, N);
            MemoryBuffer2D<float> d_c = gpu.Allocate<float>(N, N);

            d_a.CopyFrom(a, Index2.Zero, Index2.Zero, new Index2(N, N));
            d_b.CopyFrom(b, Index2.Zero, Index2.Zero, new Index2(N, N));

            matrixMulKernel(new Index2(N, N), d_a.View, d_b.View, d_c.View, N);

            // Wait for the kernel to finish...
            gpu.Synchronize();

            var c = d_c.GetAsArray();

            return c;

        }

        private static void MatrixMulShared(ArrayView2D<float> a, ArrayView2D<float> b, ArrayView2D<float> c, int N)
        {
            //Returns wrong result if groupSize > N caused by local and global indexes missmatch

            int index = Grid.GlobalIndex.X;
            if (index >= c.Length) return;

            int gx = index % N;
            int gy = index / N;

            int indexInGroup = Group.IdxX;

            int lx = indexInGroup % groupSize;
            int ly = indexInGroup / groupSize;

            float sum = 0;

            var sa = SharedMemory.Allocate2D<float>(groupSize, groupSize);
            var sb = SharedMemory.Allocate2D<float>(groupSize, groupSize);

            for (int k = 0; k < N; k += groupSize)
            {
                sa[lx, ly] = a[lx + k, gy];
                sb[lx, ly] = b[gx, ly + k];
                Group.Barrier();
                for (int r = 0; r < groupSize; r++)
                    sum += sa[r, ly] * sb[lx, r];
                Group.Barrier();
            }
            c[gy, gx] = sum;
        }


        public static float[] RunMatrixMulShared(float[,] a, float[,] b, int N)
        {
            //Create context and accelerator
            var gpu = new CudaAccelerator(new Context());

            //Create typed launcher
            var matrixMulKernelShared = gpu.LoadStreamKernel<
                ArrayView2D<float>,
                ArrayView2D<float>,
                ArrayView2D<float>,
                int>(MatrixMulShared);

            //Allocate memory
            //var bufferSize = N * N;
            MemoryBuffer2D<float> d_a = gpu.Allocate<float>(N, N);
            MemoryBuffer2D<float> d_b = gpu.Allocate<float>(N, N);
            MemoryBuffer2D<float> d_c = gpu.Allocate<float>(N, N);

            d_a.CopyFrom(a, Index2.Zero, Index2.Zero, new Index2(N, N));
            d_b.CopyFrom(b, Index2.Zero, Index2.Zero, new Index2(N, N));

            //Groups per grid dimension
            int GrPerDim = (int)Math.Ceiling((float)N / groupSize);

            KernelConfig dimension = (
                                GrPerDim * GrPerDim, // Number of groups
                                groupSize * groupSize); // Group size (thread count in group)

            matrixMulKernelShared(dimension, d_a.View, d_b.View, d_c.View, N);

            // Wait for the kernel to finish...
            gpu.Synchronize();

            var c = d_c.GetAsArray();

            return c;

        }

        private static void FloydWarshall(Index1 index, int k, ArrayView<float> d_graphMinDist, int N)
        {
            int col = index % N; //Each thread along x is assigned to a matrix column
            int row = index / N; //Each block along y is assigned to a matrix row

            float d_graphMinDist_row_k = d_graphMinDist[N * row + k];
            if (d_graphMinDist_row_k == Double.MaxValue)   //If element (row, k) = infinity, no update is needed
                return;

            float d_graphMinimumDistances_k_col = d_graphMinDist[k * N + col];
            if (d_graphMinimumDistances_k_col == Double.MaxValue)    //If element (k, col) = infinity, no update is needed
                return;

            float candidateBetterDist = d_graphMinDist_row_k + d_graphMinimumDistances_k_col;
            if (candidateBetterDist < d_graphMinDist[index])
            {
                d_graphMinDist[index] = candidateBetterDist;
            }
        }

        public static float[] RunFloydWarshall(float[,] a, int N)
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

            d_graphMinDist.CopyFrom(FlatternArr(a), 0, Index1.Zero, a.Length);

            for (int k = 0; k < N; k++)
            {
                floydWarshallKernel(bufSize, k, d_graphMinDist, N);
            }
            gpu.Synchronize();

            return d_graphMinDist.GetAsArray();

        }


    }
}
