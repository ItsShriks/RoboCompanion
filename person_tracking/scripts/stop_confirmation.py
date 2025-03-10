import speech_recognition as sr
import rospy
from std_msgs.msg import String
import time

class SpeechRecognitionService:
    _instances = {}
    
    @classmethod
    def get_instance(cls, instance_id="default", **kwargs):
        """Get a specific named instance or create it if it doesn't exist"""
        if instance_id not in cls._instances:
            cls._instances[instance_id] = cls(instance_id=instance_id, **kwargs)
        return cls._instances[instance_id]
    
    @classmethod
    def create_new_instance(cls, instance_id=None, **kwargs):
        """Always create a new instance with an optional ID"""
        if instance_id is None:
            instance_id = f"instance_{len(cls._instances)}"
        instance = cls(instance_id=instance_id, **kwargs)
        cls._instances[instance_id] = instance
        return instance
    
    def __init__(self, instance_id="default", device_part_name="Razer Barracuda X", pause_threshold=1.5, 
                 adjust_duration=20):
        self.instance_id = instance_id
        self.device_part_name = device_part_name
        
        # Initialize recognizer
        self.r = sr.Recognizer()
        self.r.pause_threshold = pause_threshold
        
        # Initialize microphone
        self.init_microphone(adjust_duration)
        
        rospy.loginfo(f"Speech Recognition Service instance '{instance_id}' initialized")
        ## For the robot to speak
        self.say_pub = rospy.Publisher('/say', String, queue_size=10)
        
        ##For interest extraction
        self.extracted_interest = None
        self.interest_received = False
        #self.detected_interest=rospy.Subscriber('ollama_detected_interest', String, self.interest_callback)
        #self.interest_input_pub = rospy.Publisher('interest_input', String, queue_size=10)
    
    def init_microphone(self, adjust_duration=20):
        # Initialize microphone
        self.mic_source = sr.Microphone()
        with self.mic_source as source:
            self.r.adjust_for_ambient_noise(source, duration=adjust_duration)
            rospy.loginfo(f"[{self.instance_id}] Microphone is set up and ambient noise level adjusted.")
            
    def say_this(self, message):
        self.say_pub.publish(message)
        rospy.loginfo(f"Saying: {message}")

    def listen_and_transcribe(self):
        rospy.loginfo(f"[{self.instance_id}] Listening...")
        with self.mic_source as source:
            audio = self.r.listen(source, timeout=6)
        try:
            # Directly use AudioData object for recognition
            recognized_text = self.r.recognize_google(audio)
            rospy.loginfo(f"[{self.instance_id}] Recognized Text: {recognized_text}")
            return recognized_text
        except (sr.UnknownValueError, sr.RequestError) as e:
            rospy.loginfo(f"[{self.instance_id}] Speech recognition error: {e}")
            return None
        
    def confirm_response(self, confirmation_not_understood_msg, not_heard_msg,try_again_msg, max_confirmation_attempts = 3):
        confirmation_attempts = 0
        confirmed = False
        while confirmation_attempts < max_confirmation_attempts:
                confirmation_attempts += 1
                confirmation_response = self.listen_and_transcribe()

                if confirmation_response:
                    confirmation_response_lower = confirmation_response.lower()
                    if any(word in confirmation_response_lower for word in ["yes", "yeah", "correct", "right", "yep"]):
                        confirmed = True
                        break
                    elif any(word in confirmation_response_lower for word in ["no", "not", "wrong", "incorrect", "nope"]):
                        self.say_this(try_again_msg)
                        confirmed = False
                        break
                    else:
                        self.say_this(confirmation_not_understood_msg)
                else:
                    self.say_this(not_heard_msg)
                    
        return confirmed
        
    def get_and_confirm_input(self, initial_prompt, validation_func = None, confirmation_format=None, 
                         failed_validation_msg=None, not_heard_msg=None,
                         confirmation_not_understood_msg=None, try_again_msg=None,
                         success_msg_format=None, max_attempts=3):
        """
        Prompts for input, validates it, then asks for confirmation before accepting.
        
        Args:
            initial_prompt: Initial question to ask the user
            validation_func: Function to validate the initial response
            confirmation_format: Format string for confirmation (default: "I heard {value}. Is that correct?")
            failed_validation_msg: Message when validation fails (default: "I could not understand that")
            not_heard_msg: Message when nothing is heard (default: "I did not hear you")
            confirmation_not_understood_msg: Message when confirmation response isn't clear
            try_again_msg: Message to say before trying again after rejection (default: "Let's try again")
            success_msg_format: Format string for success message (default: "Great!")
            max_attempts: Maximum number of attempts (None for unlimited)
            
        Returns:
            The validated and confirmed value, or None if max attempts reached
        """
        # Set default messages
        if confirmation_format is None:
            confirmation_format = "I heard {value}. Is that correct?"
        if failed_validation_msg is None:
            failed_validation_msg = "I could not understand that, could you please repeat?"
        if not_heard_msg is None:
            not_heard_msg = "I did not hear you, could you please speak louder?"
        if confirmation_not_understood_msg is None:
            confirmation_not_understood_msg = "I didn't catch that. Please say yes or no."
        if try_again_msg is None:
            try_again_msg = "I am sorry, let's try again."
        if success_msg_format is None:
            success_msg_format = "Great!"
        rospy.loginfo("Inside speech_to_text function, starting with questions")
        attempts = 0
        while attempts <= max_attempts:
            attempts += 1
            rospy.loginfo(f"[{self.instance_id}] Attempt {attempts}" + 
                         (f" of {max_attempts}" if max_attempts else ""))

            self.say_this(initial_prompt)

            response = self.listen_and_transcribe()
            rospy.loginfo(f"[{self.instance_id}] User said: {response}")

            if response:
                    confirmed = self.confirm_response(confirmation_not_understood_msg, not_heard_msg,try_again_msg)

                    if confirmed:
                        success_msg = success_msg_format.format(value=response)
                        self.say_this(success_msg)
                else:
                    self.say_this(failed_validation_msg)
            else:
                self.say_this(not_heard_msg)

            # rospy.sleep(2) 

        rospy.loginfo(f"[{self.instance_id}] Max attempts ({max_attempts}) reached without success")
        return None
    
    #def interest_callback(self, msg):
    #    """Callback function for receiving extracted interest from GPU device"""
    #    self.extracted_interest = msg.data
    #    self.interest_received = True
    #    rospy.loginfo(f"Received extracted interest: {self.extracted_interest}")
    
    #def get_and_confirm_interest(self, max_attempts=None):
    #    """
    #    Prompts for an interest, sends transcription to GPU, waits for extraction,
    #    and asks for confirmation of the extracted interest.
    #    
    #    Args:
    #        max_attempts: Maximum number of attempts (None for unlimited)
    #        
    #    Returns:
    #        The validated and confirmed interest, or None if max attempts reached
    #    """
    #    # Default messages
    #    initial_prompt = "May I know one of your interests?"
    #    confirmation_format = "I heard your interest is {value}. Is that correct?"
    #    extraction_failed_msg = "I could not identify your interest, could you please repeat it?"
    #    not_heard_msg = "I did not hear your response, could you please say it louder"
    #    confirmation_not_understood_msg = "I didn't catch that. Please say yes or no."
    #    try_again_msg = "I am sorry, let's try again."
    #    success_msg_format = "Great! {value} sounds interesting."
    #    extraction_timeout_msg = "It's taking too long to process your interest. Let's try again."
    #    
    #    attempts = 0
    #    while max_attempts is None or attempts < max_attempts:
    #        attempts += 1
    #        rospy.loginfo(f"Interest attempt {attempts}" + 
    #                     (f" of {max_attempts}" if max_attempts else ""))
    #        
    #        # Reset flag for new attempt
    #        self.interest_received = False
    #        self.extracted_interest = None
#
    #        self.say_this(initial_prompt)
    #        rospy.sleep(2)
#
    #        # Get transcription
    #        interest_response = self.listen_and_transcribe()
    #        rospy.loginfo(f"User said: {interest_response}")
#
    #        if not interest_response:
    #            self.say_this(not_heard_msg)
    #            rospy.sleep(2)
    #            continue
#
    #        # msg = String()
    #        # msg.data = interest_response
    #        self.interest_input_pub.publish(interest_response)
    #        rospy.loginfo(f"Published interest_input: {interest_response}")
    #        
    #        print("-----------------------------------------")
    #        start_time = time.time()
    #        timeout = 60
    #        
    #        while not self.interest_received and (time.time() - start_time) < timeout:
    #            rospy.sleep(0.2)
    #        
    #        if not self.interest_received:
    #            self.say_this(extraction_timeout_msg)
    #            continue
    #        
    #        if self.extracted_interest.startswith("ERROR:"):
    #            self.say_this(extraction_failed_msg)
    #            rospy.sleep(2)
    #            continue
    #            
    #        if not self.extracted_interest:
    #            self.say_this(extraction_failed_msg)
    #            rospy.sleep(2)
    #            continue
    #            
    #        confirmation_prompt = confirmation_format.format(value=self.extracted_interest)
    #        self.say_this(confirmation_prompt)
#
    #        confirmed = False
    #        confirmation_attempts = 0
    #        max_confirmation_attempts = 3 
    #        confirmed = self.confirm_response(confirmation_not_understood_msg, not_heard_msg,try_again_msg)
    #        
    #        if confirmed:
    #            success_msg = success_msg_format.format(value=self.extracted_interest)
    #            self.say_this(success_msg)
    #            return self.extracted_interest
    #            
    #    return None