# Copyright (C) <2025>  <Roni Sam Daniel Tervo>
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import sys
import os
import time
import configparser
import re # For parsing references
import webbrowser # For opening files cross-platform (basic)
import subprocess # For more control over opening files


# --- PySide6 Imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QTabWidget, QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox,
    QListWidget, QFileDialog, QMessageBox, QGroupBox, QScrollArea, QSizePolicy,
    QProgressBar, QFrame, QStackedWidget # Added QProgressBar, QFrame, QStackedWidget
)
from PySide6.QtCore import Qt, QObject, Signal, QThread, Slot, QTimer, QSize # Added QSize
from PySide6.QtGui import QPalette, QColor, QFont, QIcon, QPixmap # Added QIcon, QPixmap

# --- AI Imports ---
try:
    import google.generativeai as genai
    import google.api_core.exceptions
    gemini_imported = True
except ImportError:
    # Need to show this error early, before the main app window exists
    app_temp = QApplication.instance() # Check if already exists
    if app_temp is None:
        app_temp = QApplication(sys.argv) # Create temporary app to show message box
    QMessageBox.warning(
        None, # No parent window yet
        "Missing Library",
        "The 'google-generativeai' library is not installed. AI features will be disabled.\n"
        "Please install it using: pip install google-generativeai"
    )
    gemini_imported = False
    # No sys.exit here, allow app to run in disabled state if possible
except Exception as e:
    app_temp = QApplication.instance()
    if app_temp is None:
        app_temp = QApplication(sys.argv)
    QMessageBox.critical(None, "Import Error", f"Error importing Google AI library: {e}")
    gemini_imported = False
    sys.exit(1) # Exit if import fails critically

# --- Global Variables & Constants ---
CONFIG_FILE = 'ai_tutor_config.ini'
DEFAULT_MODEL = 'gemini-1.5-flash'
DEFAULT_LANGUAGE = 'English'
LANGUAGES = ['English', 'Finnish']
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an AI tutor. Use ONLY the provided documents to generate questions, "
    "provide hints, and evaluate answers for a student studying the material. "
    "Be concise and focus on the document content. When evaluating, start your response "
    "EXACTLY with 'Status: Correct', 'Status: Partially Correct', or 'Status: Incorrect'. "
    "When referring to a specific part of a document in your hints or explanations, "
    "include a reference tag at the end of your response in the format "
    "'Reference: [Document Index]:[Page Number]' where [Document Index] is the "
    "zero-based index of the document provided in the prompt and [Page Number] is the relevant page number."
)
REFERENCE_REGEX = re.compile(r"Reference:\s*\[?(\d+)\]?:\[?(\d+)\]?")

POINTS_CORRECT = 10
POINTS_PARTIAL = 5
STREAK_BONUS_MULTIPLIER = 2
LEVEL_UP_THRESHOLD = 50 # Score needed within a skill to level it up

# Use QColor for colors
CORRECT_COLOR = QColor("#D4EDDA") # Light Green
PARTIAL_COLOR = QColor("#FFF3CD") # Light Yellow
INCORRECT_COLOR = QColor("#F8D7DA") # Light Red
DEFAULT_FEEDBACK_COLOR = QColor("#f0f0f0") # Default background or system default
APP_BG_COLOR = "#ECEFF1" # Light grey-blue background
SKILL_BG_COLOR = "#FFFFFF" # White background for skill cards
BUTTON_COLOR = "#58CC02" # Duolingo green
BUTTON_HOVER_COLOR = "#4CAF50" # Darker green
BUTTON_TEXT_COLOR = "#FFFFFF" # White text
HEADER_COLOR = "#FFFFFF" # White header background
PROGRESS_BAR_COLOR = "#FFC107" # Amber/Yellow for progress
FONT_FAMILY = "Arial" # Or a more rounded font if available

# Gamification / Difficulty
QUESTIONS_PER_LEVEL_UP = 5 # e.g., 5 questions to increase difficulty within a skill
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

# Special skill name for combined quiz
ALL_MATERIALS_SKILL_NAME = "All Materials Review"

# ==================================
# Worker Object for AI Calls
# ==================================
class AIWorker(QObject):
    """Runs AI tasks in a separate thread."""
    # Signals to emit results or errors back to the main thread
    result_ready = Signal(str, str) # task_identifier, result_text
    error_occurred = Signal(str, str) # task_identifier, error_message
    progress_update = Signal(str) # message for status updates

    def __init__(self, model, prompt_parts, task_identifier):
        super().__init__()
        self.model = model
        self.prompt_parts = prompt_parts
        self.task_identifier = task_identifier
        self._is_running = True # Flag to allow graceful stopping

    @Slot() # Decorator to mark this method as a slot runnable by the thread
    def run(self):
        """Execute the AI call."""
        if not self._is_running or not self.model:
            self.error_occurred.emit(self.task_identifier, "AI Worker stopped or model not configured.")
            return

        try:
            self.progress_update.emit(f"Calling AI for {self.task_identifier}...")
            print("-" * 20)
            print(f"Calling AI ({self.task_identifier}):")
            # Log prompt parts carefully
            for i, part in enumerate(self.prompt_parts):
                if isinstance(part, str):
                    print(f"  Part {i} (text): {part[:150]}{'...' if len(part) > 150 else ''}")
                else: # Should be a File object from genai.upload_file
                    uri = getattr(part, 'uri', 'N/A')
                    display_name = getattr(part, 'display_name', 'Unknown File Object')
                    print(f"  Part {i} (file): Name='{display_name}', URI='{uri}'")
            print("-" * 20)

            # --- Actual Gemini API Call ---
            if not self._is_running:
                self.error_occurred.emit(self.task_identifier, "AI Worker stopped before API call.")
                return
            response = self.model.generate_content(self.prompt_parts)

            if not self._is_running:
                self.error_occurred.emit(self.task_identifier, "AI Worker stopped during API call.")
                return

            # --- Robust Response Handling ---
            response_text = None
            try:
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                    raise ValueError(f"AI request blocked due to safety ({reason}).")

                if not response.candidates:
                    finish_message = getattr(response, 'finish_message', 'No details provided.')
                    raise ValueError(f"AI returned no response candidates. Reason: {finish_message}")

                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name

                if finish_reason not in ["STOP", "MAX_TOKENS"]:
                    safety_issue = False
                    if candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            if rating.probability.name not in ["NEGLIGIBLE", "LOW"]:
                                safety_issue = True
                                raise ValueError(f"AI response may be blocked due to safety ({rating.category.name}). Finish reason: {finish_reason}")
                    if not safety_issue:
                        raise ValueError(f"AI response stopped unexpectedly. Finish reason: {finish_reason}")

                if candidate.content and candidate.content.parts:
                    response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                else:
                    if finish_reason == "STOP": response_text = ""
                    else: raise ValueError(f"AI returned no text content. Finish reason: {finish_reason}")

            except ValueError as ve:
                raise # Re-raise the ValueError
            except Exception as e:
                raise ValueError(f"Failed to parse AI response: {e}") from e

            if response_text is None:
                raise ValueError("AI returned an unparsable or empty response.")

            print(f"AI Response Received for {self.task_identifier} (truncated):", response_text[:200].strip() + "...")
            if self._is_running:
                self.result_ready.emit(self.task_identifier, response_text.strip())

        except (google.api_core.exceptions.GoogleAPIError, ConnectionError, ValueError) as e:
            print(f"Error in AI Worker ({self.task_identifier}): {e}")
            if self._is_running:
                self.error_occurred.emit(self.task_identifier, f"AI Error: {e}")
        except Exception as e:
            print(f"Unexpected Error in AI Worker ({self.task_identifier}): {e}")
            if self._is_running:
                self.error_occurred.emit(self.task_identifier, f"Unexpected Error: {e}")

    def stop(self):
        """Signals the worker to stop processing."""
        print(f"AIWorker stop() called for task: {self.task_identifier}")
        self._is_running = False

# ==================================
# Worker Object for PDF Upload
# ==================================
class PDFUploadWorker(QObject):
    """Handles PDF uploads and processing in a separate thread."""
    finished = Signal(list) # Emits list of successful genai.File objects
    error_occurred = Signal(str)
    progress_update = Signal(str) # filename being uploaded
    file_processed = Signal(str) # filename successfully processed

    def __init__(self, file_paths, basenames):
        super().__init__()
        self.file_paths = file_paths
        self.basenames = basenames
        self._is_running = True # Flag to allow graceful stopping

    @Slot()
    def run(self):
        if not self._is_running:
            self.error_occurred.emit("PDF Uploader stopped before starting.")
            return

        temp_uploaded_references = []
        try:
            print("Starting Gemini File API Upload in worker thread...")
            for i, pdf_path in enumerate(self.file_paths):
                if not self._is_running:
                    print(f"PDFUploadWorker: Stopping run loop (file index {i}).")
                    break # Check if stopped

                filename = self.basenames[i]
                print(f"-> Uploading {filename} (Index: {i})...")
                self.progress_update.emit(filename) # Signal progress

                # --- Actual Upload Call ---
                uploaded_file = genai.upload_file(path=pdf_path, display_name=filename)

                if not self._is_running: break

                # --- Wait for processing ---
                print(f"   File '{filename}' uploaded, waiting for processing...")
                time.sleep(2) # Initial short wait
                attempts = 0
                max_attempts = 10 # Increased attempts
                wait_time = 5 # Increased wait time
                while uploaded_file.state.name == 'PROCESSING' and attempts < max_attempts and self._is_running:
                    print(f"   Checking status for '{filename}' (Attempt {attempts+1}/{max_attempts}): {uploaded_file.state.name}")
                    time.sleep(wait_time)
                    if not self._is_running: break
                    try:
                        uploaded_file = genai.get_file(uploaded_file.name)
                    except Exception as get_err:
                        print(f"   Error checking status for {filename}: {get_err}. Retrying...")
                    attempts += 1

                if not self._is_running:
                    print(f"PDFUploadWorker: Stopping run loop while waiting for '{filename}' processing.")
                    if uploaded_file and uploaded_file.state.name != 'FAILED':
                        try:
                            print(f"   Attempting to delete file '{filename}' due to stop signal...")
                            genai.delete_file(uploaded_file.name)
                            print(f"   Deleted '{filename}'.")
                        except Exception as del_e:
                            print(f"   Warning: Failed deleting file '{filename}' after stop: {del_e}")
                    break # Exit the main loop

                if uploaded_file.state.name != 'ACTIVE':
                    print(f"Error: File '{filename}' failed to process after upload. State: {uploaded_file.state.name}")
                    try:
                        print(f"   Attempting to delete unprocessed/failed file '{filename}'...")
                        genai.delete_file(uploaded_file.name)
                        print(f"   Deleted '{filename}'.")
                    except Exception as del_e:
                        print(f"   Warning: Failed deleting unprocessed file {filename}: {del_e}")
                    raise google.api_core.exceptions.GoogleAPIError(f"File '{filename}' failed to become ACTIVE. Final state: {uploaded_file.state.name}")

                temp_uploaded_references.append(uploaded_file)
                self.file_processed.emit(filename) # Signal success for this file
                print(f"<- Processed '{filename}' successfully. URI: {uploaded_file.uri}")

            # --- Loop finished ---
            if self._is_running:
                if not temp_uploaded_references and self.file_paths:
                    raise ValueError("File upload process completed but resulted in no valid references.")
                print(f"PDF Upload worker finished normally. Emitting {len(temp_uploaded_references)} references.")
                self.finished.emit(temp_uploaded_references)
            else:
                print("PDF Upload worker was stopped. Not emitting 'finished' signal.")

        except (google.api_core.exceptions.GoogleAPIError, FileNotFoundError, ValueError) as e:
            print(f"Error in PDF Upload Worker: {e}")
            if self._is_running:
                self.error_occurred.emit(f"Upload/Processing Error: {e}")
        except Exception as e:
            print(f"Unexpected Error in PDF Upload Worker: {e}")
            if self._is_running:
                self.error_occurred.emit(f"Unexpected Upload Error: {e}")

    def stop(self):
        """Signals the worker to stop processing."""
        print("PDFUploadWorker stop() called.")
        self._is_running = False


# ==================================
# Custom Skill Widget
# ==================================
class SkillWidget(QFrame):
    """A widget representing a single skill (PDF) or the combined review on the path."""
    skill_selected = Signal(str) # Emits the skill name (PDF basename or special name)

    def __init__(self, skill_name, icon_path=None, is_special=False, parent=None):
        super().__init__(parent)
        self.skill_name = skill_name
        self.is_special = is_special # Flag for combined review widget
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # --- Icon ---
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # TODO: Use different icons for regular vs special skills
        icon_color = "#607D8B" if is_special else "#CFD8DC" # Different placeholder color
        default_pixmap = QPixmap(64, 64)
        default_pixmap.fill(QColor(icon_color))
        self.icon_label.setPixmap(default_pixmap)
        layout.addWidget(self.icon_label)

        # --- Skill Name ---
        self.name_label = QLabel(skill_name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(FONT_FAMILY, 12, QFont.Weight.Bold)
        self.name_label.setFont(font)
        self.name_label.setWordWrap(True)
        layout.addWidget(self.name_label)

        # --- Progress Bar (Only for regular skills) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(not is_special) # Hide for special skill
        layout.addWidget(self.progress_bar)

        # Set object name for QSS styling
        self.setObjectName("SkillWidget")
        self.progress_bar.setObjectName("SkillProgressBar")

    def mousePressEvent(self, event):
        """Emit signal when the widget is clicked."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.skill_selected.emit(self.skill_name)
        super().mousePressEvent(event)

    def set_progress(self, value):
        """Sets the progress bar value (0-100)."""
        if not self.is_special:
            self.progress_bar.setValue(min(max(0, value), 100))


# ==================================
# Skill Path Widget
# ==================================
class SkillPathWidget(QWidget):
    """Displays the available skills in a scrollable list."""
    skill_activated = Signal(str) # Emits skill name when a skill is selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Title ---
        title_label = QLabel("Your Learning Path")
        title_font = QFont(FONT_FAMILY, 18, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("margin-bottom: 20px; color: #37474F;") # Style title
        self.main_layout.addWidget(title_label)

        # --- Scroll Area for Skills ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setObjectName("SkillScrollArea")

        self.scroll_content_widget = QWidget() # Widget inside scroll area
        self.scroll_content_widget.setObjectName("SkillScrollContent")
        self.skills_layout = QVBoxLayout(self.scroll_content_widget) # Layout for skill widgets
        self.skills_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.skills_layout.setSpacing(15)

        self.scroll_area.setWidget(self.scroll_content_widget)
        self.main_layout.addWidget(self.scroll_area)

        self.skill_widgets = {} # Store skill widgets by name

    def update_skills(self, skill_names):
        """Clears and repopulates the skill path, adding the 'All Materials' option."""
        # Clear existing widgets
        for i in reversed(range(self.skills_layout.count())):
            widget = self.skills_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.skill_widgets = {}

        # Add new skill widgets for individual PDFs
        if not skill_names:
            no_skills_label = QLabel("Go to Settings to add PDF course material!")
            no_skills_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_skills_label.setStyleSheet("font-style: italic; color: #78909C; margin-top: 30px;")
            self.skills_layout.addWidget(no_skills_label)
        else:
            for name in skill_names:
                skill_widget = SkillWidget(name, is_special=False)
                skill_widget.skill_selected.connect(self.skill_activated) # Connect signal
                self.skills_layout.addWidget(skill_widget)
                self.skill_widgets[name] = skill_widget

            # Add the "All Materials Review" skill at the end
            all_materials_widget = SkillWidget(ALL_MATERIALS_SKILL_NAME, is_special=True)
            all_materials_widget.skill_selected.connect(self.skill_activated)
            self.skills_layout.addWidget(all_materials_widget)
            self.skill_widgets[ALL_MATERIALS_SKILL_NAME] = all_materials_widget

    def set_skill_progress(self, skill_name, progress_value):
         if skill_name in self.skill_widgets:
             self.skill_widgets[skill_name].set_progress(progress_value)


# ==================================
# Quiz Widget
# ==================================
class QuizWidget(QWidget):
    """Widget for displaying question, answer, feedback."""
    request_guidance = Signal()
    submit_answer = Signal(str) # Emits the answer text
    next_question = Signal()
    back_to_path = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(15) # Add spacing between elements

        # --- Back Button ---
        back_button_layout = QHBoxLayout()
        self.back_button = QPushButton("⬅️ Back to Path")
        self.back_button.setObjectName("BackButton") # For QSS
        self.back_button.clicked.connect(self.back_to_path)
        back_button_layout.addWidget(self.back_button)
        back_button_layout.addStretch()
        layout.addLayout(back_button_layout)

        # --- Skill Title ---
        self.skill_title_label = QLabel("Skill: [Skill Name]")
        skill_title_font = QFont(FONT_FAMILY, 16, QFont.Weight.Bold)
        self.skill_title_label.setFont(skill_title_font)
        self.skill_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.skill_title_label.setStyleSheet("margin-bottom: 10px; color: #37474F;")
        layout.addWidget(self.skill_title_label)

        # --- Question Display ---
        question_groupbox = QGroupBox("❓ Question")
        question_layout = QVBoxLayout(question_groupbox)
        self.question_display = QTextEdit()
        self.question_display.setReadOnly(True)
        self.question_display.setMinimumHeight(80)
        self.question_display.setObjectName("QuestionDisplay")
        question_layout.addWidget(self.question_display)
        layout.addWidget(question_groupbox)

        # --- Answer Input ---
        answer_groupbox = QGroupBox("✍️ Your Answer")
        answer_layout = QVBoxLayout(answer_groupbox)
        self.answer_entry = QTextEdit()
        self.answer_entry.setObjectName("AnswerEntry")
        answer_layout.addWidget(self.answer_entry)
        answer_groupbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(answer_groupbox, 1) # Add stretch factor

        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        self.guidance_button = QPushButton("💡 Guidance")
        self.submit_button = QPushButton("✔️ Submit")
        self.next_button = QPushButton("▶️ Next Question") # Renamed from start_quiz_button

        self.guidance_button.clicked.connect(self.request_guidance)
        self.submit_button.clicked.connect(self._emit_submit)
        self.next_button.clicked.connect(self.next_question)

        # Apply object names for QSS
        self.guidance_button.setObjectName("GuidanceButton")
        self.submit_button.setObjectName("SubmitButton")
        self.next_button.setObjectName("NextButton")

        button_layout.addWidget(self.guidance_button)
        button_layout.addWidget(self.submit_button)
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)
        layout.addLayout(button_layout)

        # --- Feedback Display ---
        feedback_groupbox = QGroupBox("💬 Feedback / Guidance")
        feedback_layout_outer = QHBoxLayout(feedback_groupbox)

        self.feedback_display = QTextEdit()
        self.feedback_display.setReadOnly(True)
        self.feedback_display.setMinimumHeight(100)
        self.feedback_display.setObjectName("FeedbackDisplay")
        self._feedback_default_palette = self.feedback_display.palette()
        feedback_layout_outer.addWidget(self.feedback_display, 1)

        ref_button_layout = QVBoxLayout()
        self.view_reference_button = QPushButton("📄 View Ref") # Shorter text
        self.view_reference_button.setObjectName("ReferenceButton")
        # self.view_reference_button.clicked.connect(...) # Connect in main app
        self.view_reference_button.setEnabled(False)
        ref_button_layout.addWidget(self.view_reference_button)
        ref_button_layout.addStretch()
        feedback_layout_outer.addLayout(ref_button_layout)

        feedback_groupbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(feedback_groupbox)

    def _emit_submit(self):
        """Helper to get text and emit submit signal."""
        answer_text = self.answer_entry.toPlainText().strip()
        self.submit_answer.emit(answer_text)

    def set_skill_title(self, title):
        # Adjust title prefix based on special name
        prefix = "Reviewing" if title == ALL_MATERIALS_SKILL_NAME else "Skill"
        self.skill_title_label.setText(f"{prefix}: {title}")

    def set_question(self, text):
        self.question_display.setPlainText(text or "")

    def get_answer(self):
        return self.answer_entry.toPlainText().strip()

    def clear_answer(self):
        self.answer_entry.clear()

    def set_answer_enabled(self, enabled):
        self.answer_entry.setEnabled(enabled)

    def set_feedback(self, text, feedback_type="default"):
        """Sets feedback text and color."""
        self.feedback_display.setPlainText(text or "")
        palette = QPalette(self.feedback_display.palette())
        color = DEFAULT_FEEDBACK_COLOR
        if feedback_type == "correct": color = CORRECT_COLOR
        elif feedback_type == "partial": color = PARTIAL_COLOR
        elif feedback_type == "incorrect": color = INCORRECT_COLOR
        palette.setColor(QPalette.ColorRole.Base, color)
        self.feedback_display.setPalette(palette)
        self.feedback_display.setAutoFillBackground(True)
        # Schedule reset if not default
        if feedback_type != "default":
            QTimer.singleShot(2500, self._reset_feedback_style)

    def _reset_feedback_style(self):
        if self.feedback_display:
            current_palette = self.feedback_display.palette()
            default_bg = self._feedback_default_palette.color(QPalette.ColorRole.Base)
            if current_palette.color(QPalette.ColorRole.Base) != default_bg:
                self.feedback_display.setPalette(self._feedback_default_palette)
                self.feedback_display.setAutoFillBackground(False)

    def set_reference_button_enabled(self, enabled):
        self.view_reference_button.setEnabled(enabled)

    def set_buttons_state(self, state):
        """Enable/disable buttons based on quiz state ('initial', 'question', 'feedback', 'processing')."""
        is_initial = state == 'initial' # Just loaded skill, no question yet
        is_question = state == 'question' # Question displayed, waiting for answer
        is_feedback = state == 'feedback' # Feedback shown after submit
        is_processing = state == 'processing' # AI call in progress

        self.guidance_button.setEnabled(is_question and not is_processing)
        self.submit_button.setEnabled(is_question and not is_processing)
        self.next_button.setEnabled((is_initial or is_feedback) and not is_processing)
        self.answer_entry.setEnabled(is_question and not is_processing)
        self.back_button.setEnabled(not is_processing) # Always disable back during processing


# ==================================
# Settings Widget (Simplified)
# ==================================
class SettingsWidget(QWidget):
    """Widget for settings, similar to the old Settings Tab."""
    settings_applied = Signal() # Signal when Apply/Save is clicked
    upload_requested = Signal() # Signal when Upload Material is clicked
    pdfs_added = Signal(list)
    pdfs_removed = Signal(list)
    pdfs_cleared = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Back Button ---
        back_button_layout = QHBoxLayout()
        self.back_button = QPushButton("⬅️ Back to Path")
        self.back_button.setObjectName("BackButton")
        # self.back_button.clicked.connect(...) # Connect in main app
        back_button_layout.addWidget(self.back_button)
        back_button_layout.addStretch()
        layout.addLayout(back_button_layout)

        # --- Title ---
        title_label = QLabel("⚙️ Settings")
        title_font = QFont(FONT_FAMILY, 18, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("margin-bottom: 20px; color: #37474F;")
        layout.addWidget(title_label)

        # --- AI Configuration ---
        ai_groupbox = QGroupBox("🤖 AI Configuration")
        ai_layout = QFormLayout(ai_groupbox)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        ai_layout.addRow("Gemini API Key:", self.api_key_input)
        self.model_input = QLineEdit(DEFAULT_MODEL)
        ai_layout.addRow("Model Name:", self.model_input)
        self.language_combo = QComboBox()
        self.language_combo.addItems(LANGUAGES)
        ai_layout.addRow("Question Language:", self.language_combo)
        self.instruction_input = QTextEdit(DEFAULT_SYSTEM_INSTRUCTION)
        self.instruction_input.setAcceptRichText(False)
        self.instruction_input.setFixedHeight(100)
        instruction_label = QLabel("System Instruction:")
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        ai_layout.addRow(instruction_label, self.instruction_input)
        self.apply_save_button = QPushButton("💾 Apply & Save Config")
        self.apply_save_button.clicked.connect(self.settings_applied)
        ai_layout.addRow(self.apply_save_button)
        layout.addWidget(ai_groupbox)

        # --- PDF Management ---
        pdf_groupbox = QGroupBox("📚 Course Material (PDFs)")
        pdf_layout = QVBoxLayout(pdf_groupbox)
        self.pdf_list_widget = QListWidget()
        self.pdf_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        pdf_layout.addWidget(self.pdf_list_widget)
        pdf_button_layout = QHBoxLayout()
        self.add_pdf_button = QPushButton("➕ Add PDF(s)")
        self.remove_pdf_button = QPushButton("➖ Remove Selected")
        self.clear_pdf_button = QPushButton("❌ Clear All")
        self.process_button = QPushButton("☁️ Upload Material")
        self.add_pdf_button.clicked.connect(self._request_add_pdfs)
        self.remove_pdf_button.clicked.connect(self._request_remove_pdfs)
        self.clear_pdf_button.clicked.connect(self._request_clear_pdfs)
        self.process_button.clicked.connect(self.upload_requested)
        pdf_button_layout.addWidget(self.add_pdf_button)
        pdf_button_layout.addWidget(self.remove_pdf_button)
        pdf_button_layout.addWidget(self.clear_pdf_button)
        pdf_button_layout.addStretch()
        pdf_button_layout.addWidget(self.process_button)
        pdf_layout.addLayout(pdf_button_layout)
        layout.addWidget(pdf_groupbox, 1)

        # --- Status Label ---
        self.status_label_settings = QLabel("Status: Initializing...")
        layout.addWidget(self.status_label_settings)

        # Apply object names for QSS
        self.apply_save_button.setObjectName("ApplyButton")
        self.add_pdf_button.setObjectName("AddPdfButton")
        self.remove_pdf_button.setObjectName("RemovePdfButton")
        self.clear_pdf_button.setObjectName("ClearPdfButton")
        self.process_button.setObjectName("UploadButton")

    def _request_add_pdfs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select PDF Files", "", "PDF Files (*.pdf)")
        if files:
            self.pdfs_added.emit(files)

    def _request_remove_pdfs(self):
        selected_items = self.pdf_list_widget.selectedItems()
        if selected_items:
            self.pdfs_removed.emit([item.text() for item in selected_items])
        else:
            QMessageBox.warning(self, "Selection Required", "Please select PDF files to remove.")

    def _request_clear_pdfs(self):
        if self.pdf_list_widget.count() > 0:
            reply = QMessageBox.question(self, "Confirm Clear",
                                         "Remove all PDF files? This clears uploaded references and resets progress.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.pdfs_cleared.emit()
        else:
            QMessageBox.information(self, "Info", "The PDF list is already empty.")

    def update_pdf_list(self, basenames):
        self.pdf_list_widget.clear()
        self.pdf_list_widget.addItems(basenames)

    def get_settings(self):
        return {
            'api_key': self.api_key_input.text(),
            'model': self.model_input.text(),
            'language': self.language_combo.currentText(),
            'instruction': self.instruction_input.toPlainText().strip() or DEFAULT_SYSTEM_INSTRUCTION
        }

    def set_settings(self, config_data):
        self.api_key_input.setText(config_data.get('api_key', ''))
        self.model_input.setText(config_data.get('model', DEFAULT_MODEL))
        self.language_combo.setCurrentText(config_data.get('language', DEFAULT_LANGUAGE))
        self.instruction_input.setPlainText(config_data.get('instruction', DEFAULT_SYSTEM_INSTRUCTION))

    def set_status(self, message):
        self.status_label_settings.setText(f"Status: {message}")

    def set_controls_enabled(self, enabled):
        """Enable/disable all controls during processing."""
        self.api_key_input.setEnabled(enabled)
        self.model_input.setEnabled(enabled)
        self.language_combo.setEnabled(enabled)
        self.instruction_input.setEnabled(enabled)
        self.apply_save_button.setEnabled(enabled)
        self.pdf_list_widget.setEnabled(enabled)
        self.add_pdf_button.setEnabled(enabled)
        self.remove_pdf_button.setEnabled(enabled)
        self.clear_pdf_button.setEnabled(enabled)
        # Process button state depends on more factors, handle separately


# ==================================
# Main Application Window (Refactored)
# ==================================
class StudyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Study Tutor Quest!")
        self.setGeometry(100, 100, 700, 800) # Adjusted size for new layout

        # --- Application State ---
        self.pdf_file_paths = []
        self.pdf_file_basenames = []
        self.uploaded_file_references = {} # Dict: {basename: genai.File}
        self.current_skill_name = None # Basename of the PDF being quizzed OR ALL_MATERIALS_SKILL_NAME
        self.current_question = None
        self.quiz_active = False # True if ANY PDFs uploaded
        self.gemini_configured = False
        self.model = None
        self.api_key = ""
        self.selected_model_name = DEFAULT_MODEL
        self.system_instruction = DEFAULT_SYSTEM_INSTRUCTION
        self.selected_language = DEFAULT_LANGUAGE
        self.current_reference = None # Parsed ref dict {'doc_index': int, 'page': int}

        # --- Gamification & Difficulty State ---
        # Progress is tracked per individual PDF skill only
        self.skill_progress = {} # Dict: {basename: {'score': 0, 'streak': 0, 'level': 1, 'questions_answered': 0, 'current_difficulty_index': 0}}
        self.total_score = 0 # Global score across all skills
        self._level_up_occurred = False # For visual cue

        # --- Threading ---
        self.thread = None
        self.worker = None

        # --- UI Elements ---
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0) # Use full window space
        self.main_layout.setSpacing(0)
        self.setCentralWidget(self.central_widget)

        self._create_header() # Stats header

        # --- Stacked Widget for Views ---
        self.view_stack = QStackedWidget()
        self.skill_path_widget = SkillPathWidget()
        self.quiz_widget = QuizWidget()
        self.settings_widget = SettingsWidget()

        self.view_stack.addWidget(self.skill_path_widget) # Index 0
        self.view_stack.addWidget(self.quiz_widget)       # Index 1
        self.view_stack.addWidget(self.settings_widget)   # Index 2

        self.main_layout.addWidget(self.view_stack)

        # --- Connect Signals ---
        self.settings_button.clicked.connect(self.show_settings_view)
        self.skill_path_widget.skill_activated.connect(self.start_quiz_for_skill) # Handles both regular and "All Materials"
        self.quiz_widget.back_to_path.connect(self.show_skill_path_view)
        self.quiz_widget.next_question.connect(self.next_question)
        self.quiz_widget.request_guidance.connect(self.get_guidance)
        self.quiz_widget.submit_answer.connect(self.submit_answer)
        self.quiz_widget.view_reference_button.clicked.connect(self.view_reference) # Connect ref button
        self.settings_widget.back_button.clicked.connect(self.show_skill_path_view)
        self.settings_widget.settings_applied.connect(self.apply_and_save_config)
        self.settings_widget.upload_requested.connect(self.process_pdfs)
        self.settings_widget.pdfs_added.connect(self.add_pdfs)
        self.settings_widget.pdfs_removed.connect(self.remove_pdfs_by_name)
        self.settings_widget.pdfs_cleared.connect(self.clear_pdfs)

        # --- Apply Stylesheet ---
        self.apply_stylesheet()

        # --- Load Config and Initial State ---
        self.load_config()
        if not self.gemini_configured:
            self.update_status_message("AI not configured. Go to Settings.")
        self.update_header_stats()
        self.update_skill_path_display() # Populate skill path initially
        self.show_skill_path_view() # Start on skill path


    def _create_header(self):
        """Creates the top header bar for stats and settings button."""
        self.header_widget = QFrame()
        self.header_widget.setObjectName("Header")
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(15, 10, 15, 10)

        # --- Stats ---
        self.level_label = QLabel("⭐ Level: 1") # Shows highest skill level
        self.score_label = QLabel("🏆 Score: 0") # Shows total score

        font = QFont(FONT_FAMILY, 11, QFont.Weight.Bold)
        self.level_label.setFont(font)
        self.score_label.setFont(font)

        header_layout.addWidget(self.level_label)
        header_layout.addSpacing(20)
        header_layout.addWidget(self.score_label)
        header_layout.addStretch()

        # --- Settings Button ---
        self.settings_button = QPushButton("⚙️") # Icon-like button
        self.settings_button.setObjectName("SettingsButton")
        self.settings_button.setFixedSize(QSize(35, 35)) # Make it square-ish
        self.settings_button.setToolTip("Settings")
        header_layout.addWidget(self.settings_button)

        self.main_layout.addWidget(self.header_widget)

    def update_header_stats(self):
        """Updates the global stats in the header."""
        self.score_label.setText(f"🏆 Score: {self.total_score}")
        # Find highest level achieved across individual skills
        max_level = 1
        if self.skill_progress:
            max_level = max(data['level'] for data in self.skill_progress.values()) if self.skill_progress else 1
        self.level_label.setText(f"⭐ Level: {max_level}")

    def apply_stylesheet(self):
        """Applies QSS styles for the Duolingo look."""
        # --- Reusing the same QSS from the provided 'new code' ---
        qss = f"""
            QMainWindow {{
                background-color: {APP_BG_COLOR};
            }}
            #Header {{
                background-color: {HEADER_COLOR};
                border-bottom: 1px solid #B0BEC5;
            }}
            #Header QLabel {{
                color: #37474F; /* Dark grey text */
            }}
            #SettingsButton {{
                font-size: 18px;
                background-color: transparent;
                border: none;
                padding: 5px;
            }}
            #SettingsButton:hover {{
                background-color: #ECEFF1; /* Light hover */
                border-radius: 5px;
            }}

            /* --- Skill Path --- */
            #SkillScrollArea {{
                border: none;
                background-color: transparent;
            }}
            #SkillScrollContent {{
                background-color: transparent;
                padding: 10px; /* Add padding around skills */
            }}
            #SkillWidget {{
                background-color: {SKILL_BG_COLOR};
                border-radius: 15px;
                border: 1px solid #CFD8DC; /* Light border */
                min-height: 150px; /* Ensure minimum size */
            }}
            #SkillWidget QLabel {{
                color: #37474F;
            }}
            #SkillProgressBar {{
                border: 1px solid #CFD8DC;
                border-radius: 5px;
                height: 10px;
                text-align: center; /* Hide text anyway */
                background-color: #ECEFF1; /* Bar background */
            }}
            #SkillProgressBar::chunk {{
                background-color: {PROGRESS_BAR_COLOR};
                border-radius: 4px;
            }}

            /* --- Quiz & Settings Widgets --- */
            QuizWidget, SettingsWidget {{
                background-color: {APP_BG_COLOR};
                padding: 15px;
            }}
            QGroupBox {{
                font-weight: bold;
                color: #37474F;
                border: 1px solid #CFD8DC;
                border-radius: 8px;
                margin-top: 10px;
                padding: 15px 10px 10px 10px; /* top, right, bottom, left */
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: #546E7A;
            }}
            QTextEdit, QLineEdit, QComboBox {{
                border: 1px solid #CFD8DC;
                border-radius: 5px;
                padding: 8px;
                background-color: {SKILL_BG_COLOR}; /* White background */
                color: #37474F;
                font-size: 11pt;
            }}
            QTextEdit:disabled, QLineEdit:disabled, QComboBox:disabled {{
                background-color: #ECEFF1; /* Slightly greyed out when disabled */
                color: #90A4AE;
            }}
            QListWidget {{
                border: 1px solid #CFD8DC;
                border-radius: 5px;
                background-color: {SKILL_BG_COLOR};
            }}

            /* --- Buttons --- */
            QPushButton {{
                background-color: #B0BEC5; /* Default greyish */
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 11pt;
                font-weight: bold;
                min-width: 80px; /* Ensure buttons have some width */
            }}
            QPushButton:hover {{
                background-color: #90A4AE;
            }}
            QPushButton:disabled {{
                background-color: #CFD8DC; /* Lighter grey when disabled */
                color: #78909C;
            }}

            /* Specific Button Styles */
            #SubmitButton, #NextButton, #UploadButton, #ApplyButton {{
                background-color: {BUTTON_COLOR};
            }}
            #SubmitButton:hover, #NextButton:hover, #UploadButton:hover, #ApplyButton:hover {{
                background-color: {BUTTON_HOVER_COLOR};
            }}

            #GuidanceButton, #ReferenceButton {{
                background-color: #1E88E5; /* Blue */
            }}
            #GuidanceButton:hover, #ReferenceButton:hover {{
                background-color: #1565C0; /* Darker Blue */
            }}

            #BackButton {{
                background-color: transparent;
                color: #546E7A;
                font-size: 10pt;
                font-weight: normal;
                border: 1px solid #CFD8DC;
                min-width: 0px;
                padding: 5px 10px;
            }}
            #BackButton:hover {{
                background-color: #ECEFF1;
                border: 1px solid #90A4AE;
            }}

            #RemovePdfButton, #ClearPdfButton {{
                background-color: #EF5350; /* Reddish */
            }}
            #RemovePdfButton:hover, #ClearPdfButton:hover {{
                background-color: #D32F2F; /* Darker Red */
            }}

            #AddPdfButton {{
                background-color: #42A5F5; /* Lighter Blue */
            }}
            #AddPdfButton:hover {{
                background-color: #1E88E5; /* Blue */
            }}

            /* Feedback Colors (set via palette, but QSS can override) */
            #FeedbackDisplay[feedbackState="correct"] {{ background-color: {CORRECT_COLOR.name()}; }}
            #FeedbackDisplay[feedbackState="partial"] {{ background-color: {PARTIAL_COLOR.name()}; }}
            #FeedbackDisplay[feedbackState="incorrect"] {{ background-color: {INCORRECT_COLOR.name()}; }}

        """
        self.setStyleSheet(qss)

    # --- View Management ---
    def show_skill_path_view(self):
        self.update_skill_path_display() # Ensure it's up-to-date
        self.view_stack.setCurrentIndex(0)
        self.current_skill_name = None # No skill active when viewing path

    def show_quiz_view(self, skill_name):
        self.current_skill_name = skill_name
        self.quiz_widget.set_skill_title(skill_name)
        self.view_stack.setCurrentIndex(1)
        # Reset quiz widget for the new skill/review
        self.quiz_widget.set_question("Click '▶️ Next Question' to start.")
        self.quiz_widget.clear_answer()
        self.quiz_widget.set_feedback("")
        self.quiz_widget.set_reference_button_enabled(False)
        self.quiz_widget.set_buttons_state('initial') # Set initial button state

    def show_settings_view(self):
        # Update settings UI with current state before showing
        self.settings_widget.set_settings({
            'api_key': self.api_key,
            'model': self.selected_model_name,
            'language': self.selected_language,
            'instruction': self.system_instruction
        })
        self.settings_widget.update_pdf_list(self.pdf_file_basenames)
        self.update_status_message() # Update status label in settings
        self.view_stack.setCurrentIndex(2)

    # --- Configuration Methods ---
    def load_config(self):
        """Loads configuration and progress from INI file."""
        config = configparser.ConfigParser()
        global DEFAULT_SYSTEM_INSTRUCTION
        if os.path.exists(CONFIG_FILE):
            try:
                config.read(CONFIG_FILE)
                self.api_key = config.get('Credentials', 'APIKey', fallback='')
                self.selected_model_name = config.get('Settings', 'Model', fallback=DEFAULT_MODEL)
                loaded_instruction = config.get('Settings', 'SystemInstruction', fallback=DEFAULT_SYSTEM_INSTRUCTION)
                self.selected_language = config.get('Settings', 'Language', fallback=DEFAULT_LANGUAGE)

                # Update default if loaded, store current value
                if loaded_instruction:
                    DEFAULT_SYSTEM_INSTRUCTION = loaded_instruction
                self.system_instruction = DEFAULT_SYSTEM_INSTRUCTION

                # Load progress data
                self.skill_progress = {} # Reset before loading
                if config.has_section('Progress'):
                    for skill_name in config['Progress']:
                        try:
                            # Parse data like: score=10,streak=2,level=1,answered=5,diff_idx=0
                            data_str = config['Progress'][skill_name]
                            # Use regex to handle potential missing keys gracefully
                            data = {}
                            for item in data_str.split(','):
                                key_val = item.split('=', 1)
                                if len(key_val) == 2:
                                    data[key_val[0].strip()] = key_val[1].strip()

                            self.skill_progress[skill_name] = {
                                'score': int(data.get('score', 0)),
                                'streak': int(data.get('streak', 0)),
                                'level': int(data.get('level', 1)),
                                'questions_answered': int(data.get('answered', 0)),
                                'current_difficulty_index': int(data.get('diff_idx', 0))
                            }
                        except Exception as e:
                            print(f"Warning: Could not parse progress for '{skill_name}': {e}")
                self._recalculate_total_score()

                if self.api_key and gemini_imported:
                    self.configure_ai() # Try configuring

            except configparser.Error as e:
                QMessageBox.warning(self, "Config Error", f"Error reading config file: {e}")
            except Exception as e:
                QMessageBox.critical(self, "Config Load Error", f"Unexpected error loading config: {e}")
        else:
            print("Config file not found. Using defaults.")
            # Ensure default values are set
            self.api_key = ""
            self.selected_model_name = DEFAULT_MODEL
            self.system_instruction = DEFAULT_SYSTEM_INSTRUCTION
            self.selected_language = DEFAULT_LANGUAGE
            self.skill_progress = {} # Reset progress if no config

        # Update settings UI elements
        self.settings_widget.set_settings({
            'api_key': self.api_key,
            'model': self.selected_model_name,
            'language': self.selected_language,
            'instruction': self.system_instruction
        })
        self.update_status_message()


    def save_config(self):
        """Saves current settings AND progress to INI file."""
        config = configparser.ConfigParser()
        # Get current settings from state variables
        config['Credentials'] = {'APIKey': self.api_key}
        config['Settings'] = {
            'Model': self.selected_model_name,
            'SystemInstruction': self.system_instruction,
            'Language': self.selected_language
        }
        # Save progress (only for individual skills)
        config['Progress'] = {}
        for skill_name, data in self.skill_progress.items():
             # Only save progress for actual PDF files, not the combined review
             if skill_name != ALL_MATERIALS_SKILL_NAME:
                config['Progress'][skill_name] = (
                    f"score={data['score']},"
                    f"streak={data['streak']},"
                    f"level={data['level']},"
                    f"answered={data['questions_answered']},"
                    f"diff_idx={data['current_difficulty_index']}"
                )

        try:
            with open(CONFIG_FILE, 'w') as configfile:
                config.write(configfile)
            print("Configuration and progress saved.")
            return True
        except IOError as e:
            QMessageBox.critical(self, "Config Save Error", f"Could not write config file '{CONFIG_FILE}': {e}")
            return False
        except Exception as e:
            QMessageBox.critical(self, "Config Save Error", f"Unexpected error saving config: {e}")
            return False

    def configure_ai(self):
        """Configures the Gemini API using current state variables."""
        if not gemini_imported:
            self.update_status_message("Error: google-generativeai library not installed.")
            self.gemini_configured = False; self.model = None
            return False

        # Get values directly from state variables
        api_key_val = self.api_key
        model_name_val = self.selected_model_name
        system_instruction_val = self.system_instruction

        if not api_key_val:
            self.update_status_message("Error: API Key is missing.")
            self.gemini_configured = False; self.model = None
            return False
        if not model_name_val:
            self.update_status_message("Error: Model Name is missing.")
            self.gemini_configured = False; self.model = None
            return False

        self.update_status_message("Configuring AI...")
        QApplication.processEvents()

        try:
            genai.configure(api_key=api_key_val)
            self.model = genai.GenerativeModel(
                model_name=model_name_val,
                system_instruction=system_instruction_val
            )
            self.gemini_configured = True
            self.update_status_message(f"AI Configured with model '{model_name_val}'.")
            return True

        except google.api_core.exceptions.PermissionDenied:
            QMessageBox.critical(self, "AI Config Error", "Permission Denied. Check API Key/Project.")
            self.update_status_message("Error: Permission Denied.")
            self.gemini_configured = False; self.model = None; return False
        except google.api_core.exceptions.NotFound:
            QMessageBox.critical(self, "AI Config Error", f"Model '{model_name_val}' not found.")
            self.update_status_message(f"Error: Model '{model_name_val}' not found.")
            self.gemini_configured = False; self.model = None; return False
        except Exception as e:
            error_message = f"Unexpected error during AI config: {e}"
            QMessageBox.critical(self, "AI Config Error", error_message)
            print(f"AI Configuration Error: {e}")
            self.update_status_message(f"Error configuring AI.")
            self.gemini_configured = False; self.model = None; return False


    def apply_and_save_config(self):
        """Applies settings from UI, saves, and reconfigures AI."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot change settings while a task is running.")
            return

        # Get settings from UI
        settings_data = self.settings_widget.get_settings()
        self.api_key = settings_data['api_key']
        self.selected_model_name = settings_data['model']
        self.system_instruction = settings_data['instruction']
        self.selected_language = settings_data['language']

        if self.save_config(): # Save updated state
            self.configure_ai() # Reconfigure AI

    # --- Settings Tab PDF Methods ---
    @Slot(list)
    def add_pdfs(self, files):
        """Adds PDF files from the signal."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot change files while a task is running.")
            return

        added_count = 0
        pdfs_changed = False
        for file_path in files:
            if file_path not in self.pdf_file_paths:
                basename = os.path.basename(file_path)
                if basename not in self.pdf_file_basenames:
                    self.pdf_file_paths.append(file_path)
                    self.pdf_file_basenames.append(basename)
                    added_count += 1
                    pdfs_changed = True
                    # Initialize progress for new skill if it doesn't exist
                    if basename not in self.skill_progress:
                        self.skill_progress[basename] = {
                            'score': 0, 'streak': 0, 'level': 1,
                            'questions_answered': 0, 'current_difficulty_index': 0
                        }
                else:
                    QMessageBox.warning(self, "Duplicate Filename", f"Skipping '{file_path}'. A file named '{basename}' is already added.")

        if pdfs_changed:
            self._clear_uploaded_references() # Clear backend refs
            self.settings_widget.update_pdf_list(self.pdf_file_basenames) # Update UI list
            self.update_skill_path_display() # Update skill path view
            self.update_status_message(f"Added {added_count} file(s). Ready to upload.")
            self.quiz_active = False # Require re-upload

    @Slot(list)
    def remove_pdfs_by_name(self, basenames_to_remove):
        """Removes selected PDF files by basename."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot change files while a task is running.")
            return

        removed_count = 0
        pdfs_changed = False
        indices_to_remove = [i for i, name in enumerate(self.pdf_file_basenames) if name in basenames_to_remove]

        if not indices_to_remove: return

        # Remove from internal lists (reverse order)
        for i in sorted(indices_to_remove, reverse=True):
            try:
                basename = self.pdf_file_basenames.pop(i)
                del self.pdf_file_paths[i]
                # Remove progress data
                if basename in self.skill_progress:
                    del self.skill_progress[basename]
                removed_count += 1
                pdfs_changed = True
            except IndexError:
                print(f"Warning: Index {i} out of sync during removal.")

        if pdfs_changed:
            self._clear_uploaded_references() # Clear backend refs
            self.settings_widget.update_pdf_list(self.pdf_file_basenames) # Update UI
            self.update_skill_path_display() # Update skill path view
            self._recalculate_total_score()
            self.update_header_stats()
            self.update_status_message(f"Removed {removed_count} PDF(s).")
            self.quiz_active = bool(self.uploaded_file_references) # Update active state


    @Slot()
    def clear_pdfs(self):
        """Removes all PDF files."""
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Cannot change files while a task is running.")
            return

        if not self.pdf_file_paths: return # Already empty

        # Confirmation is handled in SettingsWidget
        self.pdf_file_paths = []
        self.pdf_file_basenames = []
        self.skill_progress = {} # Clear all progress
        self._clear_uploaded_references() # Clear backend refs
        self.settings_widget.update_pdf_list(self.pdf_file_basenames) # Update UI
        self.update_skill_path_display() # Update skill path view
        self._recalculate_total_score()
        self.update_header_stats()
        self.update_status_message("Cleared all PDF files.")
        self.quiz_active = False

    def _clear_uploaded_references(self):
        """Clears stored file references (backend deletion is TODO)."""
        # TODO: Add backend deletion via genai.delete_file if needed
        print("Clearing uploaded file references.")
        self.uploaded_file_references = {}
        self.quiz_active = False # Quiz inactive until next successful upload

    def update_status_message(self, message=None):
        """Updates the status label in the Settings widget."""
        base_text = ""
        if message:
            status_text = base_text + message
        else:
            # Default status message based on state
            if not gemini_imported: status_text = base_text + "Error - google-generativeai library missing."
            elif not self.gemini_configured: status_text = base_text + "AI not configured. Enter API Key and Apply."
            elif not self.pdf_file_paths: status_text = base_text + "AI Configured. Add PDF files."
            elif not self.quiz_active: status_text = base_text + "Ready to upload PDF material."
            else: status_text = base_text + f"{len(self.uploaded_file_references)} file(s) uploaded. Ready for quiz ({self.selected_language})."

        self.settings_widget.set_status(status_text)
        # print(f"Settings Status Updated: {status_text}") # Debug

    def update_skill_path_display(self):
        """Updates the SkillPathWidget with current PDFs and progress."""
        self.skill_path_widget.update_skills(self.pdf_file_basenames)
        for name, data in self.skill_progress.items():
             # Calculate progress percentage based on level (max level assumed 5 for 100%)
             # Only update progress for actual skills, not the combined review
             if name != ALL_MATERIALS_SKILL_NAME:
                 max_level_for_progress = 5.0 # Assume 5 levels for 100%
                 progress_percent = min(100, (data['level'] / max_level_for_progress) * 100)
                 self.skill_path_widget.set_skill_progress(name, int(progress_percent))


    # --- PDF Upload/Processing (using QThread) ---
    def process_pdfs(self):
        """Starts the PDF upload process."""
        if not self.gemini_configured:
            QMessageBox.critical(self, "Error", "AI not configured. Set API Key in Settings.")
            self.show_settings_view()
            return
        if not self.pdf_file_paths:
            QMessageBox.critical(self, "Error", "No PDF files selected.")
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Already processing a task.")
            return

        self._set_processing_state(True, "Starting PDF upload...")
        self._clear_uploaded_references() # Clear old refs first

        # --- Setup Thread and Worker ---
        self.thread = QThread(self)
        self.worker = PDFUploadWorker(list(self.pdf_file_paths), list(self.pdf_file_basenames))
        self.worker.moveToThread(self.thread)

        # --- Connect Signals ---
        self.worker.progress_update.connect(self._handle_upload_progress)
        self.worker.file_processed.connect(self._handle_file_processed)
        self.worker.finished.connect(self._handle_upload_finished)
        self.worker.error_occurred.connect(self._handle_upload_error)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error_occurred.connect(self.thread.quit)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        print("Starting PDF Upload QThread...")
        self.thread.start()

    @Slot(str)
    def _handle_upload_progress(self, filename):
        self.update_status_message(f"Uploading {filename}...")
        QApplication.processEvents()

    @Slot(str)
    def _handle_file_processed(self, filename):
        print(f"Successfully processed in backend: {filename}")
        # Can update UI incrementally if needed

    @Slot(list)
    def _handle_upload_finished(self, uploaded_file_list):
        """Stores successful uploads in the dict."""
        print(f"PDFUploadWorker finished signal received with {len(uploaded_file_list)} files.")
        if self.worker is None or not isinstance(self.worker, PDFUploadWorker) or not self.worker._is_running:
            print("Upload finished signal received, but worker stopped or invalid.")
            self._clear_uploaded_references()
            # State is reset in _on_thread_finished
            return

        # Store references by basename
        self.uploaded_file_references = {f.display_name: f for f in uploaded_file_list if hasattr(f, 'display_name')}
        self.quiz_active = bool(self.uploaded_file_references)

        final_message = f"Successfully uploaded {len(self.uploaded_file_references)} files. Ready for quiz."
        QMessageBox.information(self, "Upload Complete", final_message)
        self.update_status_message(final_message)
        # Let _on_thread_finished handle UI state reset

    @Slot(str)
    def _handle_upload_error(self, error_message):
        print(f"Handling Upload Error signal: {error_message}")
        QMessageBox.critical(self, "Upload Error", error_message)
        self._clear_uploaded_references()
        self.update_status_message(f"Upload failed: {error_message.split(':')[0]}")
        # Let _on_thread_finished handle UI state reset

    # --- Quiz Flow Methods ---
    @Slot(str)
    def start_quiz_for_skill(self, skill_name):
        """Switches to quiz view for the selected skill or 'All Materials'."""
        if not self.quiz_active:
            QMessageBox.warning(self, "Upload Needed", "Please 'Upload Material' in Settings before starting a quiz.")
            self.show_settings_view()
            return
        # Check if all individual files needed are uploaded if it's not the special skill
        if skill_name != ALL_MATERIALS_SKILL_NAME and skill_name not in self.uploaded_file_references:
             QMessageBox.critical(self, "Error", f"'{skill_name}' has not been uploaded successfully.")
             return
        # Check if *any* files are uploaded if it *is* the special skill
        if skill_name == ALL_MATERIALS_SKILL_NAME and not self.uploaded_file_references:
            QMessageBox.critical(self, "Error", "No materials have been uploaded successfully for the review.")
            return

        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Please wait for the current task to finish.")
            return

        print(f"Starting quiz for: {skill_name}")
        self.show_quiz_view(skill_name)


    def _get_current_skill_data(self):
        """Safely gets the progress data for the current INDIVIDUAL skill."""
        # Returns default if it's the "All Materials" review or skill not found
        if self.current_skill_name and self.current_skill_name != ALL_MATERIALS_SKILL_NAME and self.current_skill_name in self.skill_progress:
            return self.skill_progress[self.current_skill_name]
        else:
            # Return default data
            return {'score': 0, 'streak': 0, 'level': 1, 'questions_answered': 0, 'current_difficulty_index': 0}

    def _update_current_skill_data(self, data):
        """Safely updates the progress data for the current INDIVIDUAL skill."""
        # Does nothing if it's the "All Materials" review
        if self.current_skill_name and self.current_skill_name != ALL_MATERIALS_SKILL_NAME:
            self.skill_progress[self.current_skill_name] = data
            self._recalculate_total_score()
            self.update_header_stats() # Update global score display
            # Update skill path progress bar after saving
            self.update_skill_path_display()
            self.save_config() # Save progress immediately

    def _recalculate_total_score(self):
        """Sums up scores from all individual skills."""
        self.total_score = sum(data['score'] for name, data in self.skill_progress.items() if name != ALL_MATERIALS_SKILL_NAME)


    def _build_document_list_for_prompt(self):
        """Creates a formatted string listing documents and their indices for the AI prompt."""
        if not self.pdf_file_basenames: return "No documents provided.\n"
        doc_list = "Documents Provided (Use these indices for references):\n"
        for i, name in enumerate(self.pdf_file_basenames):
            doc_list += f"  [{i}]: {name}\n"
        return doc_list + "\n"


    def next_question(self):
        """Handles 'Next Question' click for either single skill or all materials."""
        if not self.current_skill_name:
             QMessageBox.critical(self, "Error", "No skill or review selected.")
             return
        if not self.quiz_active or not self.uploaded_file_references:
             QMessageBox.critical(self, "Error", "Materials not uploaded successfully.")
             return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Already processing a task.")
            return

        is_all_materials_mode = (self.current_skill_name == ALL_MATERIALS_SKILL_NAME)

        # Determine difficulty (only for individual skills)
        difficulty = "Standard" # Default for all materials review
        if not is_all_materials_mode:
            skill_data = self._get_current_skill_data()
            difficulty_index = skill_data['current_difficulty_index']
            difficulty = DIFFICULTY_LEVELS[difficulty_index]

        self.current_question = None # Clear current question
        self.quiz_widget.clear_answer()
        self.quiz_widget.set_question("⏳ Generating question...")
        self.quiz_widget.set_feedback("")
        self.quiz_widget.set_reference_button_enabled(False)
        self._reset_feedback_style()

        prompt_parts = []
        file_references_for_prompt = []

        if is_all_materials_mode:
            # Use all uploaded files
            file_references_for_prompt.extend(self.uploaded_file_references.values())
            doc_list_prompt = self._build_document_list_for_prompt() # Get list with indices
            prompt_parts.append(
                f"Generate a single, clear, text-based exam question in {self.selected_language} with '{difficulty}' difficulty. "
                f"The question must test understanding of concepts, definitions, or processes explained ONLY within ANY of the provided documents. "
                f"The question should require more than a simple yes/no answer. Avoid trivial or ambiguous questions.\n\n"
                f"{doc_list_prompt}" # Include the list of documents and their indices
                f"Documents Data:\n"
            )
        else:
            # Use only the specific file for the current skill
            file_ref = self.uploaded_file_references.get(self.current_skill_name)
            if not file_ref:
                QMessageBox.critical(self, "Error", f"Could not find uploaded file data for '{self.current_skill_name}'.")
                self._set_processing_state(False) # Reset processing state manually
                return
            file_references_for_prompt.append(file_ref)
            prompt_parts.append(
                 # Use index 0 for single doc reference tag
                f"Generate a single, clear, text-based exam question in {self.selected_language} with '{difficulty}' difficulty. "
                f"The question must test understanding of concepts, definitions, or processes explained ONLY within the provided document ('{self.current_skill_name}'). "
                f"The question should require more than a simple yes/no answer. Avoid trivial or ambiguous questions.\n\n"
                f"Document Provided (Use index [0] for references):\n [0]: {self.current_skill_name}\n\n"
                f"Document Data:\n"
            )

        prompt_parts.extend(file_references_for_prompt)

        if self._start_ai_task(prompt_parts, "next_question"):
            self.quiz_widget.set_buttons_state('processing') # Disable buttons during AI call


    def get_guidance(self):
        """Handles 'Get Guidance' click."""
        if not self.current_question or not self.current_skill_name or not self.quiz_active or not self.uploaded_file_references:
            QMessageBox.warning(self, "Error", "No active question or materials.")
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Already processing a task.")
            return

        is_all_materials_mode = (self.current_skill_name == ALL_MATERIALS_SKILL_NAME)
        current_answer = self.quiz_widget.get_answer()
        prompt_parts = []
        file_references_for_prompt = []

        if is_all_materials_mode:
            file_references_for_prompt.extend(self.uploaded_file_references.values())
            doc_list_prompt = self._build_document_list_for_prompt()
            prompt_parts.append(
                f"Context: The student is answering the following question, based ONLY on the provided documents:\nQuestion: {self.current_question}\n\n"
                f"Student's current answer: \"{current_answer}\"\n\n"
                f"Task: Provide a concise hint or guiding question (1-2 sentences) in {self.selected_language} to help the student find the answer using ONLY the provided documents. "
                f"Do NOT give the answer. Focus on suggesting where to look or what concept to reconsider. "
                f"If relevant, **append a reference tag** like 'Reference: [Document Index]:[Page Number]', using the index from the document list below.\n\n"
                f"{doc_list_prompt}"
                f"Documents Data:\n"
            )
        else:
            file_ref = self.uploaded_file_references.get(self.current_skill_name)
            if not file_ref:
                QMessageBox.critical(self, "Error", f"Could not find uploaded file data for '{self.current_skill_name}'.")
                return
            file_references_for_prompt.append(file_ref)
            prompt_parts.append(
                f"Context: The student is answering the following question, based ONLY on the provided document ('{self.current_skill_name}'):\nQuestion: {self.current_question}\n\n"
                f"Student's current answer: \"{current_answer}\"\n\n"
                f"Task: Provide a concise hint or guiding question (1-2 sentences) in {self.selected_language} to help the student find the answer using ONLY the provided document. "
                f"Do NOT give the answer. Focus on suggesting where to look or what concept to reconsider. "
                f"If relevant, **append a reference tag** like 'Reference: [0]:[Page Number]'.\n\n" # Use index 0 for single doc
                f"Document Provided:\n [0]: {self.current_skill_name}\n\n"
                f"Document Data:\n"
            )

        prompt_parts.extend(file_references_for_prompt)

        if self._start_ai_task(prompt_parts, "get_guidance"):
            self.quiz_widget.set_buttons_state('processing')


    def submit_answer(self, user_answer):
        """Handles 'Submit Answer' action."""
        if not self.current_question or not self.current_skill_name or not self.quiz_active or not self.uploaded_file_references:
            QMessageBox.warning(self, "Error", "No active question or materials.")
            return
        if not user_answer:
            QMessageBox.warning(self, "Input Needed", "Please enter an answer.")
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Already processing a task.")
            return

        self.quiz_widget.set_answer_enabled(False) # Disable entry during eval

        is_all_materials_mode = (self.current_skill_name == ALL_MATERIALS_SKILL_NAME)
        prompt_parts = []
        file_references_for_prompt = []

        if is_all_materials_mode:
            file_references_for_prompt.extend(self.uploaded_file_references.values())
            doc_list_prompt = self._build_document_list_for_prompt()
            prompt_parts.append(
                f"Context: Evaluate the student's answer based ONLY on the provided documents.\n\n"
                f"Question: {self.current_question}\n"
                f"Student's Answer: {user_answer}\n\n"
                f"Task: Evaluate in {self.selected_language} using ONLY the documents:\n"
                f"1. Status: Start *EXACTLY* with 'Status: Correct', 'Status: Partially Correct', or 'Status: Incorrect'.\n"
                f"2. Correct Answer (from Docs): Provide the Correct Answer derived ONLY from the relevant document(s).\n"
                f"3. Explanation: Analyze the student's answer against the correct one. Explain errors/omissions based ONLY on the documents.\n\n"
                f"If relevant, **append a reference tag** like 'Reference: [Document Index]:[Page Number]', using the index from the list below.\n\n"
                f"{doc_list_prompt}"
                f"Documents Data:\n"
            )
        else:
            file_ref = self.uploaded_file_references.get(self.current_skill_name)
            if not file_ref:
                QMessageBox.critical(self, "Error", f"Could not find uploaded file data for '{self.current_skill_name}'.")
                self.quiz_widget.set_answer_enabled(True) # Re-enable on error
                return
            file_references_for_prompt.append(file_ref)
            prompt_parts.append(
                f"Context: Evaluate the student's answer based ONLY on the provided document ('{self.current_skill_name}').\n\n"
                f"Question: {self.current_question}\n"
                f"Student's Answer: {user_answer}\n\n"
                f"Task: Evaluate in {self.selected_language} using ONLY the document:\n"
                f"1. Status: Start *EXACTLY* with 'Status: Correct', 'Status: Partially Correct', or 'Status: Incorrect'.\n"
                f"2. Correct Answer (from Doc): Provide the Correct Answer derived ONLY from the document.\n"
                f"3. Explanation: Analyze the student's answer against the correct one. Explain errors/omissions based ONLY on the document.\n\n"
                f"If relevant, **append a reference tag** like 'Reference: [0]:[Page Number]'.\n\n" # Use index 0 for single doc
                f"Document Provided:\n [0]: {self.current_skill_name}\n\n"
                f"Document Data:\n"
            )

        prompt_parts.extend(file_references_for_prompt)

        if self._start_ai_task(prompt_parts, "submit_answer"):
            self.quiz_widget.set_buttons_state('processing')


    # --- AI Call Handling (using QThread) ---
    def _start_ai_task(self, prompt_parts, task_identifier):
        """Starts an AI task (question, guidance, submit) in the worker thread."""
        if not self.gemini_configured or self.model is None:
            QMessageBox.critical(self, "Error", "AI is not configured.")
            self._set_processing_state(False) # Ensure UI is re-enabled
            return False
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "AI is processing. Please wait.")
            return False

        # Set processing state (disables buttons)
        self._set_processing_state(True, f"AI processing: {task_identifier}...")
        if task_identifier != 'next_question': # Don't overwrite generating message
            self.quiz_widget.set_feedback("⏳ Processing AI request...")
            self._reset_feedback_style()
        QApplication.processEvents()

        # --- Setup Thread and Worker ---
        self.thread = QThread(self)
        self.worker = AIWorker(self.model, prompt_parts, task_identifier)
        self.worker.moveToThread(self.thread)

        # --- Connect Signals ---
        self.worker.result_ready.connect(self._handle_ai_result)
        self.worker.error_occurred.connect(self._handle_ai_error)
        self.worker.progress_update.connect(self._handle_ai_progress)
        self.thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self.thread.quit)
        self.worker.error_occurred.connect(self.thread.quit)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        return True


    @Slot(str, str) # task_identifier, result_text
    def _handle_ai_result(self, task_identifier, result_text):
        """Handles successful results from the AI worker."""
        print(f"AI Result received for: {task_identifier}")
        if self.worker is None or not isinstance(self.worker, AIWorker) or not self.worker._is_running:
            print("Warning: AI result received, but worker stopped or invalid.")
            # State is reset in _on_thread_finished
            return

        is_all_materials_mode = (self.current_skill_name == ALL_MATERIALS_SKILL_NAME)

        if task_identifier == "next_question":
            self.current_question = result_text
            self.quiz_widget.set_question(self.current_question)
            difficulty_text = ""
            if not is_all_materials_mode:
                 difficulty_text = f" ({DIFFICULTY_LEVELS[self._get_current_skill_data()['current_difficulty_index']]})"
            self.quiz_widget.set_feedback(f"New question{difficulty_text}. Enter your answer.")
            self.quiz_widget.set_buttons_state('question') # Enable answer/submit/guidance
            self.quiz_widget.answer_entry.setFocus()

        elif task_identifier == "get_guidance":
            self.display_feedback(f"Hint:\n{result_text}") # Handles ref parsing
            self.quiz_widget.set_buttons_state('question') # Keep in question state

        elif task_identifier == "submit_answer":
            feedback_lower = result_text.lower()
            feedback_type = "default"
            self._level_up_occurred = False # Reset visual cue flag

            points_earned = 0
            streak_broken = False

            if feedback_lower.startswith("status: correct"):
                feedback_type = "correct"
                points_earned = POINTS_CORRECT
                if not is_all_materials_mode: # Only update streak/bonus for individual skills
                    skill_data = self._get_current_skill_data()
                    skill_data['streak'] += 1
                    bonus = skill_data['streak'] * STREAK_BONUS_MULTIPLIER
                    points_earned += bonus
                    print(f"Correct! +{POINTS_CORRECT} points, +{bonus} streak bonus.")
                else:
                    print(f"Correct! +{POINTS_CORRECT} points (Review Mode).")

            elif feedback_lower.startswith("status: partially correct"):
                feedback_type = "partial"
                points_earned = POINTS_PARTIAL
                streak_broken = True
                print(f"Partially Correct. +{POINTS_PARTIAL} points. Streak reset.")
            elif feedback_lower.startswith("status: incorrect"):
                feedback_type = "incorrect"
                streak_broken = True
                print("Incorrect. Streak reset.")
            else:
                print("Warning: AI feedback status unclear. No score change.")
                feedback_type = "default"
                streak_broken = True # Treat unclear as incorrect for streak

            # --- Update skill data ONLY if NOT in "All Materials" mode ---
            if not is_all_materials_mode:
                skill_data = self._get_current_skill_data()
                skill_data['score'] += points_earned
                skill_data['questions_answered'] += 1
                if streak_broken:
                    skill_data['streak'] = 0

                # Difficulty / Level Up Logic (only for individual skills)
                answered_count = skill_data['questions_answered']
                current_diff_idx = skill_data['current_difficulty_index']
                if answered_count > 0 and answered_count % QUESTIONS_PER_LEVEL_UP == 0 and points_earned > 0: # Level up difficulty on correct answer after N questions
                    if current_diff_idx < len(DIFFICULTY_LEVELS) - 1:
                        skill_data['current_difficulty_index'] += 1
                        print(f"Difficulty increased to {DIFFICULTY_LEVELS[skill_data['current_difficulty_index']]} for {self.current_skill_name}")

                # Level up based on score within the skill
                new_level = 1 + (skill_data['score'] // LEVEL_UP_THRESHOLD)
                if new_level > skill_data['level']:
                    print(f"Skill Level Up! '{self.current_skill_name}' reached Level {new_level}")
                    skill_data['level'] = new_level
                    self._level_up_occurred = True

                # Save updated skill data and update header
                self._update_current_skill_data(skill_data)
            else:
                # If in "All Materials" mode, maybe update total score directly? Or just don't track.
                # Let's not track progress for "All Materials" for now.
                pass

            # Display feedback (handles ref parsing and color)
            self.display_feedback(result_text, feedback_type=feedback_type)

            # Reset for next question
            self.current_question = None
            self.quiz_widget.set_buttons_state('feedback') # Show feedback, enable 'Next'

        else:
            print(f"Unknown AI task identifier in result handler: {task_identifier}")
            self.quiz_widget.set_feedback(f"Result for unknown task '{task_identifier}':\n{result_text}")
            # Try to recover state
            if self.current_question:
                self.quiz_widget.set_buttons_state('question')
            else:
                self.quiz_widget.set_buttons_state('initial')


    @Slot(str, str) # task_identifier, error_message
    def _handle_ai_error(self, task_identifier, error_message):
        """Handles errors reported by the AI worker."""
        print(f"Handling AI Error signal ({task_identifier}): {error_message}")
        if self.worker is None or not isinstance(self.worker, AIWorker):
            print("Warning: AI error received, but worker invalid.")
            return # State reset in _on_thread_finished

        QMessageBox.critical(self, f"AI Error ({task_identifier})", error_message)
        self.display_feedback(f"Error during '{task_identifier}': {error_message}", feedback_type="incorrect")

        # Reset UI state depending on failed task (buttons re-enabled in _on_thread_finished)
        if task_identifier == "next_question":
            self.current_question = None
            self.quiz_widget.set_question("Error generating question. Click 'Next' to try again.")
            self.quiz_widget.set_buttons_state('initial') # Allow retry
        elif task_identifier == "get_guidance":
            # Question still active, allow retry
            self.quiz_widget.set_buttons_state('question')
        elif task_identifier == "submit_answer":
            # Question still active, answer still there, allow retry
            self.quiz_widget.set_answer_enabled(True)
            self.quiz_widget.set_buttons_state('question')


    @Slot(str) # status_update message
    def _handle_ai_progress(self, status_update):
        """Handles progress messages from the AI worker (optional)."""
        pass # Currently not used for UI updates


    # --- UI State Management ---
    def _set_processing_state(self, processing: bool, status_message: str = None):
        """Enable/disable UI elements based on processing state."""
        is_actively_processing = processing or (self.thread is not None and self.thread.isRunning())

        # Disable Settings controls during any processing
        self.settings_widget.set_controls_enabled(not is_actively_processing)
        # Settings Button in header
        self.settings_button.setEnabled(not is_actively_processing)

        # Quiz buttons are handled by quiz_widget.set_buttons_state based on AI call results/errors
        # But ensure back button is disabled during processing
        self.quiz_widget.back_button.setEnabled(not is_actively_processing)
        self.settings_widget.back_button.setEnabled(not is_actively_processing)

        # Special handling for Upload button
        can_process_pdfs = not is_actively_processing and self.gemini_configured and bool(self.pdf_file_paths)
        self.settings_widget.process_button.setEnabled(can_process_pdfs)

        # Update status label in settings
        if status_message:
            self.update_status_message(status_message)
        elif not is_actively_processing: # If processing just finished, update default status
            self.update_status_message()

        QApplication.processEvents()

    def display_feedback(self, text, feedback_type="default"):
        """Displays feedback in quiz view, parses refs, sets color, manages ref button."""
        self.current_reference = None # Reset before parsing
        self.quiz_widget.set_reference_button_enabled(False) # Disable initially
        self.quiz_widget.set_feedback(text, feedback_type) # Set text and color

        # --- Parse Reference ---
        # Reference format from AI should be "Reference: [Index]:[Page Number]"
        # Index refers to the list provided in the prompt (either single doc [0] or list index)
        match = REFERENCE_REGEX.search(text)
        if match:
            try:
                doc_index_from_ai = int(match.group(1))
                page_num = int(match.group(2))

                is_all_materials_mode = (self.current_skill_name == ALL_MATERIALS_SKILL_NAME)

                # Determine the actual index in the main self.pdf_file_paths list
                actual_doc_index = -1
                if is_all_materials_mode:
                    # AI gives index based on the full list provided in the prompt
                    actual_doc_index = doc_index_from_ai
                else:
                    # AI gives index [0], map it to the current skill's index in the main list
                    if self.current_skill_name in self.pdf_file_basenames:
                        actual_doc_index = self.pdf_file_basenames.index(self.current_skill_name)

                # Validate the actual index
                if 0 <= actual_doc_index < len(self.pdf_file_paths):
                    self.current_reference = {'doc_index': actual_doc_index, 'page': page_num}
                    self.quiz_widget.set_reference_button_enabled(True) # Enable button
                    print(f"Found reference: Mapped Index={actual_doc_index} ('{self.pdf_file_basenames[actual_doc_index]}'), Page={page_num}")
                else:
                    print(f"Warning: Found ref tag but calculated Doc Index {actual_doc_index} is out of range.")

            except (ValueError, IndexError) as e:
                print(f"Error parsing reference tag: {match.group(0)} - {e}")

    def _reset_feedback_style(self):
        """Resets the feedback display style in the quiz widget."""
        self.quiz_widget._reset_feedback_style()


    # --- Reference Viewing Method ---
    def view_reference(self):
        """Opens the referenced PDF file (uses index from main pdf list)."""
        if not self.current_reference:
            QMessageBox.information(self, "No Reference", "No valid document reference found.")
            return

        doc_index = self.current_reference.get('doc_index') # Index in the main pdf_file_paths list
        page_num = self.current_reference.get('page')

        if doc_index is None or not (0 <= doc_index < len(self.pdf_file_paths)):
            QMessageBox.critical(self, "Reference Error", f"Invalid document index ({doc_index}).")
            return

        file_path = self.pdf_file_paths[doc_index]
        basename = self.pdf_file_basenames[doc_index]

        if not os.path.exists(file_path):
            QMessageBox.critical(self, "File Not Found", f"Cannot find '{basename}' at:\n{file_path}")
            return

        print(f"Opening '{basename}' (Path: {file_path}) at page {page_num if page_num else 'N/A'}")
        try:
            if sys.platform == "win32": os.startfile(file_path)
            elif sys.platform == "darwin": subprocess.run(["open", file_path], check=True)
            else: subprocess.run(["xdg-open", file_path], check=True)
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Error opening '{basename}': {e}")
            # Try fallback
            try:
                abs_path = os.path.abspath(file_path)
                # Handle spaces and potential special characters better for file URI
                import urllib.parse
                file_uri = urllib.parse.urljoin('file:', urllib.request.pathname2url(abs_path))
                webbrowser.open(file_uri)
            except Exception as wb_err:
                QMessageBox.critical(self, "Open Error", f"Fallback webbrowser open failed: {wb_err}")


    # --- Thread Finish Handling ---
    @Slot()
    def _on_thread_finished(self):
        """Handles cleanup AFTER a QThread finishes."""
        print(f"QThread finished signal received. Clearing Python references.") # Debug
        # Store references before clearing
        worker_type = type(self.worker).__name__ if self.worker else "None"
        task_id = self.worker.task_identifier if hasattr(self.worker, 'task_identifier') else "N/A"

        self.thread = None
        self.worker = None
        # Update UI state now that processing is confirmed done
        print(f"--> Calling _set_processing_state(False) from _on_thread_finished (Worker: {worker_type}, Task: {task_id})") # Debug
        self._set_processing_state(False)

        # Ensure quiz buttons reflect the correct state IF a quiz was active
        # This needs to happen *after* _set_processing_state potentially re-enables things
        if self.view_stack.currentIndex() == 1: # If on quiz view
            # Check the state based on whether a question is loaded or feedback was shown
            if self.current_question:
                 # If an AI call failed mid-question, we might end up here with a question loaded
                 # Ensure buttons are in 'question' state
                 self.quiz_widget.set_buttons_state('question')
            elif self.quiz_widget.feedback_display.toPlainText():
                 # Feedback was shown (likely after submit_answer success/error)
                 self.quiz_widget.set_buttons_state('feedback')
            else:
                 # No question, no feedback (likely error during next_question)
                 self.quiz_widget.set_buttons_state('initial')


    # --- Window Closing ---
    def closeEvent(self, event):
        """Handle window close event, stop threads, save config."""
        print("Close event triggered.")
        if self.thread is not None and self.thread.isRunning():
            reply = QMessageBox.question(self, "Confirm Exit",
                                         "A task is running. Quit anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                print("Stopping worker and thread...")
                if self.worker: self.worker.stop()
                self.thread.quit()
                if not self.thread.wait(1500):
                    print("Thread termination required.")
                    self.thread.terminate()
                    self.thread.wait(500)
            else:
                print("Close event ignored.")
                event.ignore()
                return # Don't save if user cancels

        # Save config/progress on normal close or after stopping thread
        print("Saving configuration before exiting...")
        self.save_config()
        print("Accepting close event.")
        event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    # os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1' # Optional scaling
    app = QApplication(sys.argv)
    window = StudyApp()
    window.show()
    sys.exit(app.exec())
