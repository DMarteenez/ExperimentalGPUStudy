using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using Cudafy.Types;

namespace ExperimentalGPUStudy
{
    class CUDAFYProc
    {
        [Cudafy]
        private static void thekernel()
        {
        }

        public static void RunTest()
        {
            CudafyModule km = CudafyTranslator.Cudafy();
            GPGPU gpu = CudafyHost.GetDevice(eGPUType.Cuda);
            gpu.LoadModule(km);


            gpu.Launch().thekernel(); // or gpu.Launch(1, 1, "kernel"); 
            Console.WriteLine("Sample kernel started successfully!");
        }
    }
}
