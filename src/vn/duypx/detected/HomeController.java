package vn.duypx.detected;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import jssc.SerialPort;
import jssc.SerialPortEvent;
import jssc.SerialPortEventListener;
import jssc.SerialPortException;
import jssc.SerialPortList;

public class HomeController {
	@FXML
	private Button startCamBtn;
	@FXML
	private ImageView currentFrame;
	@FXML
	private Label statusLabel;

	private boolean cameraActive = false;
	private static int cameraId = 0;

	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that realizes the video capture
	private VideoCapture capture;
	// a flag to change the button behavior

	private byte[] graphDef;
	private List<String> labels;

	// getting serial ports list into the array
	String[] portNames = SerialPortList.getPortNames();
	private static SerialPort serialPort;

	protected void init() {
		this.capture = new VideoCapture();
	}

	@FXML
	protected void startCamera(ActionEvent event) {
		getModel();
		if (!this.cameraActive) {
			// start the video capture
			this.capture.open(cameraId);

			// is the video stream available?
			if (this.capture.isOpened()) {
				this.cameraActive = true;

				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {

					@Override
					public void run() {
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(currentFrame, imageToShow);
					}
				};

				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

				// update the button content
				this.startCamBtn.setText("Stop Camera");
			} else {
				// log the error
				System.err.println("Impossible to open the camera connection...");
			}
		} else {
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.startCamBtn.setText("Start Camera");

			// stop the timer
			this.stopAcquisition();
		}
	}

	@FXML
	protected void takeAPhoto(ActionEvent event) {
		captureImage();
	}

	public void captureImage() {
		String f = "check.jpg";
		Mat matrix = grabFrame();
		Imgcodecs.imwrite(f, matrix);

		byte[] imageBytes = readAllBytesOrExit(Paths.get(f));

		try (Tensor image = Tensor.create(imageBytes)) {
			float[] labelProbabilities = executeInceptionGraph(graphDef, image);
			int bestLabelIdx = maxIndex(labelProbabilities);
			switch (labels.get(bestLabelIdx)) {
			case "paper towel":
			case "candle": {
				statusLabel.setText("OK !!!");
				sendStringToCom("+z");
				break;
			}
			case "lotion": {
				statusLabel.setText("NG !!!");
				sendStringToCom("-z");
				break;
			}
			default: {
				statusLabel.setText("Detecting...");
				break;
			}
			}
			System.out.println(String.format("BEST MATCH: %s (%.2f%% likely)", labels.get(bestLabelIdx),
					labelProbabilities[bestLabelIdx] * 100f));
		} catch (Exception e) {
			statusLabel.setText("ERROR: " + e.toString());
		}
	}

	private Mat grabFrame() {
		Mat frame = new Mat();

		if (this.capture.isOpened()) {
			try {
				// read the current frame
				this.capture.read(frame);

				// if the frame is not empty, process it
//	                if (!frame.empty()) {
//	                	
//	                }

			} catch (Exception e) {
				// log the error
				System.err.println("Exception during the image elaboration: " + e);
			}
		}

		return frame;
	}

	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition() {
		if (this.timer != null && !this.timer.isShutdown()) {
			try {
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			} catch (InterruptedException e) {
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}

		if (this.capture.isOpened()) {
			// release the camera
			this.capture.release();
		}
	}

	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 *
	 * @param view  the {@link ImageView} to update
	 * @param image the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image) {
		Utils.onFXThread(view.imageProperty(), image);
	}

	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed() {
		this.stopAcquisition();
	}

	private void getModel() {
		Path modelPath = Paths.get("tensorflow_inception_graph.pb");
		graphDef = readAllBytesOrExit(modelPath);
		Path modePathTxt = Paths.get("imagenet_comp_graph_label_strings.txt");
		labels = readAllLinesOrExit(modePathTxt);
	}

	private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
		try (Graph g = new Graph()) {
			g.importGraphDef(graphDef);
			try (Session s = new Session(g);
					Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
				final long[] rshape = result.shape();
				if (result.numDimensions() != 2 || rshape[0] != 1) {
					throw new RuntimeException(String.format(
							"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
							Arrays.toString(rshape)));
				}
				int nlabels = (int) rshape[1];
				return result.copyTo(new float[1][nlabels])[0];
			}
		}
	}

	private static int maxIndex(float[] probabilities) {
		int best = 0;
		for (int i = 1; i < probabilities.length; ++i) {
			if (probabilities[i] > probabilities[best]) {
				best = i;
			}
		}
		return best;
	}

	private static byte[] readAllBytesOrExit(Path path) {
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(1);
		}
		return null;
	}

	private static List<String> readAllLinesOrExit(Path path) {
		try {
			return Files.readAllLines(path, Charset.forName("UTF-8"));
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			System.exit(0);
		}
		return null;
	}

	private void sendStringToCom(String mes) {
		if (portNames.length == 0) {
			statusLabel.setText("Not Found COM !!!");
			System.out.println(
					"There are no serial-ports :( You can use an emulator, such ad VSPE, to create a virtual serial port.");
			System.out.println("Press Enter to exit...");
			return;
		}

		for (int i = 0; i < portNames.length; i++) {
			serialPort = new SerialPort("COM1");
			try {
				serialPort.openPort();

				serialPort.setParams(SerialPort.BAUDRATE_9600, SerialPort.DATABITS_8, SerialPort.STOPBITS_1,
						SerialPort.PARITY_NONE);

				serialPort.setFlowControlMode(SerialPort.FLOWCONTROL_RTSCTS_IN | SerialPort.FLOWCONTROL_RTSCTS_OUT);

				serialPort.addEventListener(new PortReader(), SerialPort.MASK_RXCHAR);

				serialPort.writeString(mes);
			} catch (SerialPortException ex) {
				System.out.println("There are an error on writing string to port т: " + ex);
			}
		}

	}

	private static class PortReader implements SerialPortEventListener {

		@Override
		public void serialEvent(SerialPortEvent event) {
			if (event.isRXCHAR() && event.getEventValue() > 0) {
				try {
					String receivedData = serialPort.readString(event.getEventValue());
					System.out.println("Received response: " + receivedData);
				} catch (SerialPortException ex) {
					System.out.println("Error in receiving string from COM-port: " + ex);
				}
			}
		}

	}
}
