FROM tensorflow/tensorflow:latest-gpu

LABEL Maintainer = "SamH99"

WORKDIR /app

COPY /app .

# DO NOT USE FOR TRAINING!
# DOES NOT BIND MOUNT THE DIRECTORY TO THE CONTAINER!


# BUILD IMAGE
# $ docker build -t lofi .

# RUN AND SSH TO CONTAINER
# $ docker run --rm --name lofi_container -it lofi