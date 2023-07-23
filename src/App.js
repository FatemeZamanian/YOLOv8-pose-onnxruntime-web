import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage, detectVideo } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const videoRef = useRef(null);
  const [streaming, setStreaming] = useState(null); // streaming state

  const onClickVideoStream = () => {
    const height = modelInputShape[2]
    const width = modelInputShape[3]
    let video = document.getElementById('vid');
    console.log(video.width, video.height)
    let src = new cv.Mat(height, width, cv.CV_8UC4);
    let dst = new cv.Mat(height, width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(videoRef);


    const FPS = 30;
    function processVideo() {
      
        try {
          // if (!streaming) {
          //     // clean and stop.
          //     debugger
          //     src.delete();
          //     dst.delete();
          //     return;
          // }
          let begin = Date.now();
          // start processing.
          debugger
          cap.read(src);
          cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
          cv.imshow('canvasOutput', dst);
          debugger
          // schedule the next one.
          let delay = 1000/FPS - (Date.now() - begin);
          setTimeout(processVideo, delay);
      } catch (err) {
          alert(err);
      }
};

// schedule the first one.
setTimeout(processVideo, 0);


    // if not streaming
    // if (streaming === null || streaming === "image") {
    // closing image streaming
    // getVideo();
    // webcam.open(cameraRef.current); // open webcam
    // cameraRef.current.style.display = "block"; // show camera
    // setStreaming("camera"); // set streaming to camera
    // }
    // closing video streaming
    // else if (streaming === "camera") {
    // webcam.close(cameraRef.current);
    // cameraRef.current.style.display = "none";
    // videoRef
    // webcam.open(videoRef.current);
    // setStreaming(null);
    // } else alert(`Can't handle more than 1 stream\nCurrently streaming : ${streaming}`); // if streaming video
  }

  const onLoadVideo = () => {
    detectVideo(videoRef.current,
      canvasRef.current,
      session,
      topk,
      iouThreshold,
      scoreThreshold,
      modelInputShape)
  }

  // Configs
  const modelName = "yolov8n-pose.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // create session
    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`, // url
      ["Loading YOLOv8 Pose model", setLoading] // logger
    );

    const yolov8 = await InferenceSession.create(arrBufNet);

    const arrBufNMS = await download(
      `${baseModelURL}/modified_nms-yolov8.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    );
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );

    await yolov8.run({ images: tensor });


    setSession({ net: yolov8, nms: nms });
    setLoading(null);

  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>YOLOv8 Pose Detection App</h1>
        <p>
          YOLOv8 pose detection application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            );
          }}
        />

        <video width={640} height={640} id="vid" ref={videoRef} autoPlay playsInline muted style={{ display: image ? "none" : "block" }}
          onPlaying={() => detectVideo(videoRef.current,
            canvasRef.current,
            session,
            topk,
            iouThreshold,
            scoreThreshold,
            modelInputShape)} />

        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      {/* <input  ref={cameraRef} accept="video/*" style={{ display: "none" }} /> */}
      {/* <input ref={cameraRef} type="file" accept="video/*" id="capture" capture="camcorder"/> */}

      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
        <button
          onClick={onClickVideoStream}
          
        >
          {streaming === "camera" ? "Close" : "Open"} Webcam
        </button>


      </div>
    </div>
  );
};

export default App;
