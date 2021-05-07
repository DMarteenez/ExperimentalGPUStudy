using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ExperimentalGPUStudy
{
    class Program
    {
        static void PrintMatrix(float[,] a)
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

        static void PrintMatrix(float[] a, int N)
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    Console.Write(a[i * N + j] + " ");
                }
                Console.WriteLine();
            }
        }

            static float[,] GetRandomMatrix(int m, int n)
        {
            var a = new float[m, n];
            Random rnd = new Random();
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    a[i, j] = (float)rnd.NextDouble();
                }
            }
            return a;
        }

        static float[,] GetRandomMatrixInt(int m, int n)
        {
            var a = new float[m, n];
            Random rnd = new Random();
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    a[i, j] = rnd.Next(0, 10);
                }
            }
            return a;
        }

        static void Main(string[] args)
        {
            const int N = 2;

            Stopwatch sw = new Stopwatch();
            var a = GetRandomMatrixInt(N, N);
            Thread.Sleep(100);
            var b = GetRandomMatrixInt(N, N);

            PrintMatrix(a);
            Console.WriteLine();
            PrintMatrix(b);
            Console.WriteLine();

            sw.Start();
            //PrintMatrix(ILGPUProc.RunMatrixMul(a, b, N));
            PrintMatrix(ILGPUProc.RunMatrixMulShared(a, b, N), N);
            //Console.WriteLine();
            //CPUProc.FloydWarshall(a);
            //PrintMatrix(a);
            sw.Stop();
            Console.WriteLine();
            Console.WriteLine("Time = " + sw.ElapsedMilliseconds);
            


            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
