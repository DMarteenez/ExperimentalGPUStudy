using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ExperimentalGPUStudy
{
    class CPUProc
    {
        public static float[,] MatrixMul(float[,] a, float[,] b)
        {
            
            var N = a.GetLength(1);
            if (N != b.GetLength(0)) throw new Exception(message: "Wrong input matrix sizes.");

            float[,] c = new float[N, N];

            Parallel.For(0, N * N, (i) =>
            { 
                c[i / N, i % N] = 0;
                for (int r = 0; r < N; r++)
                {
                    c[i / N, i % N] += a[i / N, r] * b[r, i % N];
                }

            });

            return c;
        }

        public static void FloydWarshall(float[,] w)
        {
            var N = w.GetLength(0);
            Parallel.For(0, N, (k) =>
            {
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        w[i, j] = Math.Min(w[i, j], w[i, k] + w[k, j]);
                    }
                }
            });
        }
    }
}
