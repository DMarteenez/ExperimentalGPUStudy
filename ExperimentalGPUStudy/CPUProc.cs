using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ExperimentalGPUStudy
{
    class CPUProc
    {
        public static double[,] MatrixMul(double [,] a, double[,] b)
        {
            var l = a.GetLength(0);
            var n = b.GetLength(1);
            var m = a.GetLength(1);
            if (m != b.GetLength(0)) throw new Exception(message: "Wrong input matrix sizes.");

            double[,] c = new double[l, n];

            Parallel.For(0, l, (i) =>
            {
                for (int j = 0; j < n; j++)
                {
                    c[i, j] = 0;
                    for (int r = 0; r < m; r++)
                    {
                        c[i, j] += a[i, r] * b[r, j];
                    }
                }
            });

            return c;
        }

    }
}
