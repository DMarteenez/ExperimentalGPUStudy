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

        public static void RunOddEvenSort(ref float[] a)
        {
            int N = a.Length;
            int evenArrModifier = 1 - (N % 2);

            bool stopFlag = false;
            bool iterationEven = true;

            while (true)
            {
                if (stopFlag)
                    break;
                stopFlag = true;

                if (iterationEven)
                {
                    for(int i = 0; i < N; i += 2)
                    {
                        //swap
                        if(a[i] > a[i + 1])
                        {
                            var tmp = a[i];
                            a[i] = a[i + 1];
                            a[i + 1] = tmp;
                            stopFlag = false;
                        }
                    }
                }
                else
                {
                    for (int i = 1; i < N - evenArrModifier; i += 2)
                    {
                        //swap
                        if (a[i] > a[i + 1])
                        {
                            var tmp = a[i];
                            a[i] = a[i + 1];
                            a[i + 1] = tmp;
                            stopFlag = false;
                        }
                    }
                }
                iterationEven = !iterationEven;
            }
        }
    }
}
