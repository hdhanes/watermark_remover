# Watermark Remover

An interactive python script that trained a Pix2Pix GAN for the task of artificial watermark removal on photos.

## Organization

This folder contains all auxiliary files and code created for this project.

The validation set data is included in the nonwatermark_images and watermark_images folders.

The generated image output from each of the three models is included in its respective folder.

The trained models are kept in the models folder (only the generator for Pix2Pix). 

The scripts folder contains the main script (which contains the data prep, GAN implementation and training) as well as an auxiliary files for the auto encoders.

Lastly, the other folder contains a font file that is used to generate watermarks. This needs to be uploaded in the 4th cell of the main script.

Please email hendrix.hanes@gmail.com if there are any questions.
