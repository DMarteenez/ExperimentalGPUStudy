using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ExperimentalGPUStudy
{
    class CPUProc
    {
        public static float[][] RunMatrixMul(float[][] a, float[][] b)
        {       
            var N = a.Length;

            float[][] c = new float[N][];
            for (int i = 0; i < N; i++)
                c[i] = new float[N];

            Parallel.For(0, N * N, (i) =>
            { 
                float sum = 0;
                for (int r = 0; r < N; r++)
                {
                    sum += a[i / N][r] * b[r][i % N];
                }
                c[i / N][i % N] = sum;
            });
            return c;
        }

        public static void RunFloydWarshall(float[][] w)
        {
            var N = w.GetLength(0);
            Parallel.For(0, N, (k) =>
            {
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        w[i][j] = Math.Min(w[i][j], w[i][k] + w[k][j]);
                    }
                }
            });
        }
    }
}
