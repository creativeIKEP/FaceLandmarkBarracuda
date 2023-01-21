using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

namespace MediaPipe.FaceLandmark
{

    //
    // Face landmark detector class
    //
    public sealed class FaceLandmarkDetector : System.IDisposable
    {
        #region Public accessors

        public const int VertexCount = 468;
        public const int EyeVetrexCount = 71;
        public const int IrisVetrexCount = 5;
        public const int lipsVetrexCount = 80;

        public ComputeBuffer VertexBuffer
          => _postBuffer;

        public IEnumerable<Vector4> VertexArray
          => _postRead ? _postReadCache : UpdatePostReadCache();

        public ComputeBuffer FaceFlag;
        public ComputeBuffer LeftEye;
        public ComputeBuffer LeftIris;
        public ComputeBuffer RightEye;
        public ComputeBuffer RightIris;
        public ComputeBuffer Lips;

        #endregion

        #region Public methods

        public FaceLandmarkDetector(ResourceSet resources)
        {
            _resources = resources;
            AllocateObjects();
        }

        public void Dispose()
          => DeallocateObjects();

        public void ProcessImage(Texture image)
          => RunModel(image);

        #endregion

        #region Compile-time constants

        // Input image size (defined by the model)
        const int ImageSize = 192;

        #endregion

        #region Private objects

        ResourceSet _resources;
        ComputeBuffer _preBuffer;
        ComputeBuffer _postBuffer;
        IWorker _worker;

        void AllocateObjects()
        {
            var model = ModelLoader.Load(_resources.model);
            _preBuffer = new ComputeBuffer(ImageSize * ImageSize * 3, sizeof(float));
            _postBuffer = new ComputeBuffer(VertexCount, sizeof(float) * 4);
            _worker = model.CreateWorker();

            FaceFlag = new ComputeBuffer(1, sizeof(float));
            LeftEye = new ComputeBuffer(EyeVetrexCount, sizeof(float) * 2);
            LeftIris = new ComputeBuffer(IrisVetrexCount, sizeof(float) * 2);
            RightEye = new ComputeBuffer(EyeVetrexCount, sizeof(float) * 2);
            RightIris = new ComputeBuffer(IrisVetrexCount, sizeof(float) * 2);
            Lips = new ComputeBuffer(lipsVetrexCount, sizeof(float) * 2);
        }

        void DeallocateObjects()
        {
            _preBuffer?.Dispose();
            _preBuffer = null;

            _postBuffer?.Dispose();
            _postBuffer = null;

            _worker?.Dispose();
            _worker = null;

            FaceFlag?.Dispose();
            LeftEye?.Dispose();
            LeftIris?.Dispose();
            RightEye?.Dispose();
            RightIris?.Dispose();
            Lips?.Dispose();
        }

        #endregion

        #region Neural network inference function

        void RunModel(Texture source)
        {
            // Preprocessing
            var pre = _resources.preprocess;
            pre.SetTexture(0, "_Texture", source);
            pre.SetBuffer(0, "_Tensor", _preBuffer);
            pre.Dispatch(0, ImageSize / 8, ImageSize / 8, 1);

            // Run the BlazeFace model.
            using (var tensor = new Tensor(1, ImageSize, ImageSize, 3, _preBuffer))
                _worker.Execute(tensor);

            // Postprocessing
            var post = _resources.postprocess;

            var tempRT = _worker.CopyOutputToTempRT("output_mesh_identity", 1, VertexCount * 3);
            post.SetTexture(0, "_Tensor", tempRT);
            post.SetBuffer(0, "_Vertices", _postBuffer);
            post.Dispatch(0, VertexCount / 52, 1, 1);
            RenderTexture.ReleaseTemporary(tempRT);

            FaceFlag = TensorToBuffer("conv_faceflag", 1);

            var tempRT_LE = _worker.CopyOutputToTempRT("output_left_eye", 1, EyeVetrexCount * 2);
            post.SetInt("_TargetVertexCount", EyeVetrexCount);
            post.SetTexture(1, "_Tensor", tempRT_LE);
            post.SetBuffer(1, "_Vertices_F2", LeftEye);
            post.Dispatch(1, EyeVetrexCount, 1, 1);
            RenderTexture.ReleaseTemporary(tempRT_LE);

            var tempRT_RE = _worker.CopyOutputToTempRT("output_right_eye", 1, EyeVetrexCount * 2);
            post.SetInt("_TargetVertexCount", EyeVetrexCount);
            post.SetTexture(1, "_Tensor", tempRT_RE);
            post.SetBuffer(1, "_Vertices_F2", RightEye);
            post.Dispatch(1, EyeVetrexCount, 1, 1);
            RenderTexture.ReleaseTemporary(tempRT_RE);

            var tempRT_LI = _worker.CopyOutputToTempRT("output_left_iris", 1, IrisVetrexCount * 2);
            post.SetInt("_TargetVertexCount", IrisVetrexCount);
            post.SetTexture(1, "_Tensor", tempRT_LI);
            post.SetBuffer(1, "_Vertices_F2", LeftIris);
            post.Dispatch(1, IrisVetrexCount, 1, 1);
            RenderTexture.ReleaseTemporary(tempRT_LI);

            var tempRT_RI = _worker.CopyOutputToTempRT("output_right_iris", 1, IrisVetrexCount * 2);
            post.SetInt("_TargetVertexCount", IrisVetrexCount);
            post.SetTexture(1, "_Tensor", tempRT_RI);
            post.SetBuffer(1, "_Vertices_F2", RightIris);
            post.Dispatch(1, IrisVetrexCount, 1, 1);
            RenderTexture.ReleaseTemporary(tempRT_RI);

            var tempRT_Lip = _worker.CopyOutputToTempRT("output_lips", 1, lipsVetrexCount * 2);
            post.SetInt("_TargetVertexCount", lipsVetrexCount);
            post.SetTexture(1, "_Tensor", tempRT_Lip);
            post.SetBuffer(1, "_Vertices_F2", Lips);
            post.Dispatch(1, lipsVetrexCount, 1, 1);
            RenderTexture.ReleaseTemporary(tempRT_Lip);

            // Read cache invalidation
            _postRead = false;
        }

        #endregion

        #region GPU to CPU readback

        Vector4[] _postReadCache = new Vector4[VertexCount];
        bool _postRead;

        Vector4[] UpdatePostReadCache()
        {
            _postBuffer.GetData(_postReadCache, 0, 0, VertexCount);
            _postRead = true;
            return _postReadCache;
        }

        #endregion
    }

} // namespace MediaPipe.FaceLandmark
