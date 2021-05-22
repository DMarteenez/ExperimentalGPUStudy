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
        static void PrintMatrix(float[][] a)
        {
            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < a.GetLength(1); j++)
                {
                    Console.Write(a[i][j] + " ");
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

        static float[][] GetRandomMatrix(int m, int n)
        {
            var a = new float[m][];
            for (int i = 0; i < m; i++)
                a[i] = new float[n];  
            Random rnd = new Random();
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    a[i][j] = (float)rnd.NextDouble(); //rnd.Next(0, 10);
                }
            }
            return a;
        }

        static float[][] GetRandomMatrixInt(int m, int n)
        {
            var a = new float[m][];
            for(int i =0; i < m; i++)
                a[i] = new float[n];
            Random rnd = new Random();
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    a[i][j] = rnd.Next(0, 10);
                }
            }
            return a;
        }

        static void Main(string[] args)
        {
            int launchIter = 10;
            int warmupLaunches = 0;

            //const int N = 256;
            //const int N = 512;
            //const int N = 1024;
            //const int N = 1536;
            //const int N = 2048;
            //const int N = 3072;
            const int N = 4096;

            List<float> time = new List<float>();
                
            Stopwatch sw = new Stopwatch();
            for(int i = 0; i < launchIter; i++)
            {
                var a = GetRandomMatrix(N, N);
                Thread.Sleep(100);
                var b = GetRandomMatrix(N, N);

                sw.Restart();
                //CPUProc.RunMatrixMul(a, b);
                //CPUProc.RunFloydWarshall(a);
                ILGPUProc.RunMatrixMul(a, b, N);
                //LGPUProc.RunMatrixMulShared(a, b, N);
                //ILGPUProc.RunFloydWarshall(a, N);
                sw.Stop();

                time.Add(sw.ElapsedMilliseconds);
            }

            for(int i = 0; i < warmupLaunches; i++)
                time.RemoveAt(0);

            float sumTime = 0;
            foreach(var el in time)
            {
                sumTime += el;
            }

            Console.WriteLine();
            Console.WriteLine("Time = " + sumTime / (launchIter - warmupLaunches));

            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
