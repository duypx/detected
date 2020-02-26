package vn.duypx.detected;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.GridPane;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

import org.opencv.core.Core;

public class Main extends Application {

	@Override
	public void start(Stage primaryStage) throws Exception {
		try {
            // load the FXML resource
            FXMLLoader loader = new FXMLLoader(getClass().getResource("home_controller.fxml"));
            // store the root element so that the controllers can use it
            GridPane rootElement = (GridPane) loader.load();
            // create and style a scene
            Scene scene = new Scene(rootElement, 800, 800);
            scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
            // create the stage with the given title and the previously created
            // scene
            primaryStage.setTitle("Detected Program");
            primaryStage.setScene(scene);
            // show the GUI
            primaryStage.show();

            // set the proper behavior on closing the application
            HomeController homeController = loader.getController();
            homeController.init();
            primaryStage.setOnCloseRequest((new EventHandler<WindowEvent>() {
                public void handle(WindowEvent we) {
                    homeController.setClosed();
                }
            }));

            scene.setOnKeyPressed(new EventHandler<KeyEvent>() {
                @Override
                public void handle(KeyEvent keyEvent) {
                    System.out.println(keyEvent.getCode());
                    if (keyEvent.getCode() == KeyCode.CLOSE_BRACKET) {
                        homeController.captureImage();
                    }
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
		
	}

	public static void main(String[] args) {
        // load the native OpenCV library
		System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        launch(args);
    }
}
