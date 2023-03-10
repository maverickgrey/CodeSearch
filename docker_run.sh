#!/bin/sh

# build a image
# echo "what is the version?"
# read version
# docker build -t maverickgrey/codesearch:${version} .

# run a image
echo "running a docker image..."
read version
docker run -itd -p 8004:8000 --name codesearch -v /home/fdse/projects/DataBackUps/data:/projects/CodeSearchBackEnd/data maverickgrey/codesearch:${version}