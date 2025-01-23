FROM ood:v1.2

ADD . /
WORKDIR /

#RUN unzip /mnt/ood/CLIP-main.zip
#RUN python3 CLIP-main/setup.py install
#RUN pip install torch==1.10.1+rocm4.0.1 torchvision==0.10.2+rocm4.0.1 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
#RUN conda install -c pytorch faiss-cpu
#RUN pip install loguru
#RUN pip install tqdm
#RUN pip uninstall Pillow
#RUN pip install Pillow==6.2.2

EXPOSE 5016
ENTRYPOINT ["/root/miniconda3/bin/python","api.py"]
