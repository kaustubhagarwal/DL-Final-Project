# Deep Synthesis: Real-time Audio-Visual Synthesis Using Deep Learning

This repository contains the code for Deep Synthesis, a system that generates real-time audio-visual speech synthesis. It works by taking three inputs:

Text: The text that you want the person in the image or video to say.
Voice sample: A short audio clip of the target speaker's voice. This is used to capture the speaker's unique vocal characteristics, such as pitch, timbre, and accent.
Image or video: An image or video of the person you want to synthesize speech for. Deep Synthesis will use this image or video to create a new video where the person appears to be speaking the text with synchronized lip movements.
The system works by first using a text-to-speech (TTS) model to generate a speech waveform based on the input text and the voice sample. Then, a lip synchronization model is used to generate realistic lip movements for the person in the image or video that are synchronized with the speech waveform. Finally, the generated speech waveform and lip movements are combined to create a new video.

## Generating Samples
To generate samples run the iPynb Sample_Generator.ipynb. It is a relatively straightforward inferencing file.
Run the first few sections to get the models and necessary libraries.

Following that you can give the 3 inputs described above. Due to limited time and scope, we were not able to develop a webpage to interactively upload these inputs (something which we would've really liked to do) but we were able to give very clear directions to do so.

You will be prompted to provide the text input which you can give by typing.

The other two inputs- image and voice sample should be added to the temporary folder sample_data created on Google Colab with the names mentioned below
- The audio file should be audio_file.mp3
- The image should be img_sample.jpg

Following this you'll have to run the next 2 segments of code to generate the audio of the voice sample speaking the text provided followed by lip syncing the words in the audio sample overlayed on the image provided.

Finally, you'll be able to see the generated video on the Colab Notebook. It can also be found in the temporary files on Colab with the name 'result.mp4' which can be downloaded and used as you'd like!

This is the final project for the course ECE-GY 7123: Deep Learning by Professor Chinmay Hedge.

## Team members
- Kaustubh Agarwal (ka3210)
- Srujana Kanchisamudram (sk11115)
- Aashir Saroya (as17888)

## Acknowledgement
We express our deepest gratitude to Professor Chinmay Hedge and the TAs along with all contributors who offered their expertise and insights throughout the "Deep Synthesis" project. We acknowledge the assistance of OpenAI's language model, ChatGPT 4.0, for its role in generating parts of this report. Additionally, we are grateful for the extensive resources available through online platforms such as Stack Overflow and various GitHub repositories. Our project has also benefited greatly from the official documentation of PyTorch, NumPy, and seaborn, which guided our development decisions. Moreover, we thank the NYU High Performance Computing (HPC) facilities for the computational resources that were essential for training our models. These tools and supports were instrumental in the successful completion of our research.
