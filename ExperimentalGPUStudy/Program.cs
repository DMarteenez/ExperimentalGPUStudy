using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ExperimentalGPUStudy
{
    class Program
    {
        static void PrintMatrix(double[,] a)
        {
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    Console.Write(a[i,j] + " ");
                }
                Console.WriteLine();
            }
        }

        static double[,] GetRandomMatrix(int m, int n)
        {
            var a = new double[m, n];
            Random rnd = new Random();
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    a[i, j] = rnd.NextDouble();
                }
            }
            return a;
        }

        static void Main(string[] args)
        {
            const int N = 1000;

            Stopwatch sw = new Stopwatch();
            var a = GetRandomMatrix(N, N);
            var b = GetRandomMatrix(N, N);

            sw.Start();
            CPUProc.MatrixMul(a, b);
            sw.Stop();
            Console.WriteLine(sw.ElapsedMilliseconds);



            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
