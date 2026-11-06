"""Main PySide6 application window."""

from __future__ import annotations

from pathlib import Path

try:
    from PySide6.QtCore import QThread, Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
        QProgressBar, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout,
        QWidget, QCheckBox,
    )
    from ui.workers import PredictionWorker, TrainingWorker
except ImportError as exc:
    raise RuntimeError("PySide6 is required to run the desktop application.") from exc

from feedback import FeedbackStore
from paths import model_dir
from image_validation import DETECTION_THRESHOLD, ImageValidationError

class DropLabel(QLabel):
    def __init__(self, on_path):
        super().__init__("Drop a chest X-ray here or use Browse")
        self.on_path = on_path
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(180)
        self.setStyleSheet("border: 2px dashed #5b7cfa; border-radius: 12px; padding: 24px;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.on_path(urls[0].toLocalFile())


class MainWindow(QMainWindow):
    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[1]
        configured_model_path = Path(model_path) if model_path else model_dir() / "best.keras"
        self.model_path = (
            configured_model_path
            if configured_model_path.is_absolute()
            else self.project_root / configured_model_path
        )
        self.image_path = ""
        self.thread = None
        self.worker = None
        self.training_thread = None
        self.training_worker = None
        self.feedback_store = FeedbackStore()
        self.latest_rows = []
        self.chart_layout = QVBoxLayout()
        self.dataset_path = self.project_root / "data"
        self.setWindowTitle("Chest X-ray 17-Pathology Classifier")
        self.resize(900, 700)

        self.drop_area = DropLabel(self.set_image)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse)
        self.predict_button = QPushButton("Predict")
        self.model_available = self.model_path.is_file()
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict)
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model)
        self.dataset_button = QPushButton("Choose Data Folder")
        self.dataset_button.clicked.connect(self.choose_dataset)
        self.fine_tune_button = QPushButton("Fine-tune Feedback")
        self.fine_tune_button.clicked.connect(self.fine_tune)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        self.preview = QLabel("No image selected")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(220)
        self.results = QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(["Pathology", "Confidence", "Threshold", "Status"])
        self.model_status = QLabel()
        self.dataset_status = QLabel("Dataset: not selected")
        self.status = QLabel("Research decision-support tool; not a medical diagnosis.")
        self.findings_notice = QLabel()
        self.findings_notice.hide()
        self.update_model_status()
        chart_widget = QWidget()
        chart_widget.setLayout(self.chart_layout)
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setWidget(chart_widget)
        if self.dataset_path.is_dir():
            self.dataset_status.setText(f"Dataset: {self.dataset_path}")

        controls = QHBoxLayout()
        controls.addWidget(self.browse_button)
        controls.addWidget(self.predict_button)
        controls.addWidget(self.dataset_button)
        controls.addWidget(self.train_button)
        controls.addWidget(self.fine_tune_button)
        controls.addWidget(self.clear_button)
        layout = QVBoxLayout()
        layout.addWidget(self.drop_area)
        layout.addWidget(self.preview)
        layout.addLayout(controls)
        layout.addWidget(self.model_status)
        layout.addWidget(self.dataset_status)
        layout.addWidget(self.findings_notice)
        layout.addWidget(chart_scroll)
        layout.addWidget(self.results)
        layout.addWidget(self.status)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select chest X-ray", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.set_image(path)

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Select prepared data directory")
        if path:
            self.dataset_path = Path(path)
            self.dataset_status.setText(f"Dataset: {self.dataset_path}")


    def set_image(self, path: str):
        if not Path(path).is_file() or Path(path).suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            QMessageBox.warning(self, "Invalid image", "Select an existing PNG, JPEG, or BMP image.")
            return
        self.image_path = path
        self.preview.setPixmap(QPixmap(path).scaled(500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.predict_button.setEnabled(self.model_available)
        if self.model_available:
            self.status.setText(f"Loaded: {Path(path).name}")
        else:
            self.status.setText("Image loaded. Train or load a model before predicting.")

    def predict(self):
        if not self.image_path:
            return
        if not self.model_available:
            self.status.setText("Prediction unavailable: no trained model is loaded.")
            QMessageBox.information(
                self,
                "Model required",
                f"No trained model was found at:\n{self.model_path}\n\n"
                "Train the model first, then restart the application.",
            )
            return
        self.predict_button.setEnabled(False)
        self.thread = QThread(self)
        self.worker = PredictionWorker(self.model_path, self.image_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.show_results)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(lambda: self.predict_button.setEnabled(True))
        self.thread.start()

    def train_model(self):
        if self.training_thread is not None and self.training_thread.isRunning():
            return
        answer = QMessageBox.question(
            self,
            "Train model",
            "Training downloads the dataset and may take a long time. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.train_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self.status.setText("Loading dataset and starting training...")
        self.training_thread = QThread(self)
        self.training_worker = TrainingWorker(
            str(self.model_path.parent),
            10,
            str(self.dataset_path),
        )
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self.training_progress)
        self.training_worker.finished.connect(self.training_finished)
        self.training_worker.error.connect(self.training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.error.connect(self.training_thread.quit)
        self.training_thread.finished.connect(lambda: self.train_button.setEnabled(True))
        self.training_thread.start()

    def fine_tune(self):
        if not self.model_available:
            QMessageBox.information(self, "Model required", "Train a model before fine-tuning feedback.")
            return
        self.train_button.setEnabled(False)
        self.fine_tune_button.setEnabled(False)
        self.status.setText("Fine-tuning on clinician feedback...")
        self.training_thread = QThread(self)
        self.training_worker = TrainingWorker(str(self.model_path.parent), 1, fine_tune=True)
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self.training_progress)
        self.training_worker.finished.connect(self.training_finished)
        self.training_worker.error.connect(self.training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.error.connect(self.training_thread.quit)
        self.training_thread.finished.connect(lambda: self.fine_tune_button.setEnabled(True))
        self.training_thread.start()

    def training_progress(self, epoch: int, total: int, logs: dict):
        auc = logs.get("val_auc")
        metric = f", validation AUC: {auc:.3f}" if isinstance(auc, (int, float)) else ""
        self.status.setText(f"Training epoch {epoch}/{total}{metric}")

    def training_finished(self, _history):
        self.model_available = self.model_path.is_file()
        self.update_model_status()
        self.predict_button.setEnabled(bool(self.image_path) and self.model_available)
        self.status.setText("Training complete. Select an image and click Predict.")

    def training_failed(self, message: str):
        self.status.setText("Training failed.")
        QMessageBox.critical(
            self,
            "Training failed",
            f"{message}\n\nCheck Kaggle credentials and dataset configuration.",
        )

    def show_results(self, rows):
        self.latest_rows = rows
        has_detection = any(float(item["probability"]) >= DETECTION_THRESHOLD for item in rows)
        if has_detection:
            self.findings_notice.hide()
        else:
            self.findings_notice.setText(
                "No signs of any of the 17 supported pathologies, including tumor or "
                "cancer-related findings, were detected by the model.<br><br>"
                "<b>This is an automated result, not a medical diagnosis. The model can "
                "miss findings, so please have a doctor review the scan.</b>"
            )
            self.findings_notice.setStyleSheet(
                "background:#d9f3e6; color:#14532d; padding:10px; border-radius:6px;"
            )
            self.findings_notice.show()
        while self.chart_layout.count():
            item = self.chart_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        corrections = []
        for item in sorted(rows, key=lambda row: row["probability"], reverse=True):
            line = QHBoxLayout()
            label = QLabel(str(item["label"]))
            label.setMinimumWidth(180)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(round(float(item["probability"]) * 100))
            bar.setFormat(item["confidence"])
            color = "#e85d5d" if item["positive"] else "#4c9f70"
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            check = QCheckBox("Confirmed")
            check.setChecked(bool(item["positive"]))
            corrections.append((item["label"], check))
            line.addWidget(label)
            line.addWidget(bar, 1)
            line.addWidget(check)
            row_widget = QWidget()
            row_widget.setLayout(line)
            self.chart_layout.addWidget(row_widget)
        self.chart_layout.addStretch()
        self.correction_checks = corrections
        self.results.setRowCount(len(rows))
        for row, item in enumerate(rows):
            values = [item["label"], item["confidence"], f'{item["threshold"]:.2f}', "Positive" if item["positive"] else "Negative"]
            for column, value in enumerate(values):
                self.results.setItem(row, column, QTableWidgetItem(str(value)))
        self.status.setText("Prediction complete.")
        if not hasattr(self, "feedback_button"):
            self.feedback_button = QPushButton("Save Clinician Correction")
            self.feedback_button.clicked.connect(self.save_feedback)
            self.centralWidget().layout().insertWidget(6, self.feedback_button)

    def save_feedback(self):
        if not self.image_path or not self.latest_rows:
            return
        labels = [label for label, check in self.correction_checks if check.isChecked()]
        probabilities = [float(item["probability"]) for item in self.latest_rows]
        self.feedback_store.add(self.image_path, probabilities, labels)
        self.status.setText("Clinician correction saved for background fine-tuning.")

    def show_error(self, message: str):
        if message.startswith("This image doesn't look like"):
            QMessageBox.warning(self, "Invalid chest X-ray", message)
            self.status.setText("Invalid image rejected.")
        else:
            QMessageBox.critical(self, "Prediction failed", message)
            self.status.setText("Prediction failed.")

    def clear(self):
        self.image_path = ""
        self.preview.clear()
        self.preview.setText("No image selected")
        self.results.setRowCount(0)
        self.findings_notice.hide()
        self.predict_button.setEnabled(False)

    def closeEvent(self, event):
        if self.training_thread is not None and self.training_thread.isRunning():
            QMessageBox.warning(
                self,
                "Training in progress",
                "Wait for training to finish before closing the application.",
            )
            event.ignore()
            return
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            if not self.thread.wait(5000):
                event.ignore()
                return
        event.accept()

    def update_model_status(self):
        if self.model_available:
            self.model_status.setText(f"Model loaded: {self.model_path}")
            self.model_status.setStyleSheet("color: #55c878;")
        else:
            self.model_status.setText(
                f"No trained model found: {self.model_path}. Train the model before prediction."
            )
            self.model_status.setStyleSheet("color: #f0ad4e;")
