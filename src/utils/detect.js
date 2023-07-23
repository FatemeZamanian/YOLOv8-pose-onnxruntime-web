import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape,
  callback = () => { },
  isVideo
) => {
  // debugger
  const [modelWidth, modelHeight] = inputShape.slice(3);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight, isVideo);

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor(
    "float32",
    new Float32Array([
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
    ])
  ); // nms config tensor
  console.time("session")
  const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
  const { selected } = await session.nms.run({ detection: output0, config: config }); // perform nms and filter boxes

  const boxes = [];

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    const box = data.slice(0, 4);
    const score = data.slice(4, 5); // classes probability scores
    const landmarks = data.slice(5); // maximum probability scores
    const label = 0; // class id of maximum probability scores

    const [x, y, w, h] = [
      (box[0] - 0.5 * box[2]) * xRatio, // upscale left
      (box[1] - 0.5 * box[3]) * yRatio, // upscale top
      box[2] * xRatio, // upscale width
      box[3] * yRatio, // upscale height
    ]; // keep boxes in maxSize range

    // console.log(landmarks);
    boxes.push({
      label: label,
      probability: score,
      bounding: [x, y, w, h], // upscale box
      landmarks: landmarks
    }); // update boxes to draw later
  }
  console.timeEnd("session")
  renderBoxes(canvas, boxes, xRatio, yRatio); // Draw boxes
  callback();
  input.delete(); // delete unused Mat
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight, isVideo) => {
  // debugger
  const mat = isVideo ? source : cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};


//detect video

/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv8 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */

export const detectVideo = async (
  vidSource,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const ctx = canvas.getContext("2d");
  ctx.save();
  const detectFrame = async () => {
    const ctx = canvas.getContext("2d");
    const width = inputShape[2]
    const height = inputShape[3]
    const src = new cv.Mat(width, height, cv.CV_8UC4);
    const dst = new cv.Mat(width, height, cv.CV_8UC1);
    const FPS = 30;

    // let begin = Date.now();
    ctx.drawImage(vidSource, 0, 0, width, height);

    src.data.set(ctx.getImageData(0, 0, width, height).data);
    cv.cvtColor(src, dst, cv.COLOR_BGR2BGRA);

    // cv.imshow(canvas, dst); // canvasOutput is the id of another <canvas>;
    // schedule next one.
    // let delay = 1000 / FPS - (Date.now() - begin);
    detectImage(dst,
      canvas,
      session,
      topk,
      iouThreshold,
      scoreThreshold,
      inputShape,
      () => {
        requestAnimationFrame(detectFrame); // get another frame
      },
      true);
    // ctx.clearRect(0, 0, width, height);
    // ctx.drawImage(vidSource, 0, 0, width, height);
    // const img = new Image();
    // debugger
    // const setImageBlob = (blob) => {
    //   img.src = URL.createObjectURL(blob);
    //   debugger
    // debugger

    // detectImage(vidSource,
    //   canvas,
    //   session,
    //   topk,
    //   iouThreshold,
    //   scoreThreshold,
    //   inputShape,
    //   () => {
    //     requestAnimationFrame(detectFrame); // get another frame
    //   });
    // img.onload = () => {
    // const test5 = ctx.getImageData(0, 0, inputShape[0], inputShape[1]).data
    // debugger
    // no longer need to read the blob so it's revoked
    // };

    // img.src = url;
    // }

    // canvas.toBlob(setImageBlob);
    // const playImage = new Image();
    // playImage.src = ""
    // playImage.onload = () => {
    // canvas.height = vidSource.videoHeight;
    // canvas.width = vidSource.videoWidth;
    // const startX = vidSource.videoWidth / 2 - playImage.width / 2;
    // const startY = vidSource.videoHeight / 2 - playImage.height / 2;
    // canvas
    //   .getContext("2d")
    //   .drawImage(playImage, startX, startY, playImage.width, playImage.height);
    // canvas.toBlob() = (blob) => {
    //   const img = new Image();
    //   img.src = window.URL.createObjectUrl(blob);
    // detectImage(vidSource,
    //   canvas,
    //   session,
    //   topk,
    //   iouThreshold,
    //   scoreThreshold,
    //   inputShape,
    //   () => {
    //     requestAnimationFrame(detectFrame); // get another frame
    //   });
    // };
    // };


    // const img = canvas.toDataURL("image/png")
    // ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
    // const img = ctx.drawImage(vidSource, 0, 0)
    // debugger
    // detectImage(img,
    //   canvas,
    //   session,
    //   topk,
    //   iouThreshold,
    //   scoreThreshold,
    //   inputShape, () => {
    //     requestAnimationFrame(detectFrame); // get another frame
    //   });
  };

  // setInterval(() => {
  detectFrame(); // initialize to detect every frame
  // }, 1)
};


    // const ctx = canvas.getContext("2d");
    // const width = inputShape[0]
    // const height = inputShape[1]
    // const src = new cv.Mat(width, height, cv.CV_8UC4);
    // const dst = new cv.Mat(width, height, cv.CV_8UC1);
    // const FPS = 30;

    // function processVideo() {
    //   let begin = Date.now();
    //   ctx.drawImage(vidSource, 0, 0, width, height);

    //   src.data.set(ctx.getImageData(0, 0, width, height).data);
    //   cv.cvtColor(src, dst, cv.COLOR_BGR2BGRA);

    //   // cv.imshow(canvas, dst); // canvasOutput is the id of another <canvas>;
    //   // schedule next one.
    //   let delay = 1000 / FPS - (Date.now() - begin);
    //   setTimeout(processVideo, delay);
    // }
    // setTimeout(processVideo, 0);