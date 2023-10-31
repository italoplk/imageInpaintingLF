import socket
import argparse
import sys
from os.path import isfile
import time
from struct import unpack
import imageio

import numpy as np
from models.mlp import MLP
import torch
from dataset import normalize, denormalize

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='XXX')
    # File paths, etc.
    parser.add_argument('--address', type=str, default='127.0.0.1', 
                        help='IPv4 address where the server will listen to')
    parser.add_argument('--port', type=int, default=8080,
                        help='UDP port where the server will listen to')
    parser.add_argument('--model', type=str, default='/path/to/model', 
                        help='Path to the trained neural network')
    parser.add_argument('--model-type', type=str, default='mlp',
                        help='Type of model [mlp|graph] (default conv)')
    parser.add_argument('--context-size', type=int, default=64,
                        help='Size of the context [64, 128])')
    parser.add_argument('--predictor-size', type=int, default=32,
                        help='Size of the predictor (default 32))')
    parser.add_argument('--n-lines', type=int, default=3)
    parser.add_argument('--bit-depth', type=int, default='8', 
                        help='Pixel depth in bpp [8,10] in LE format (default 8)')
    parser.add_argument('--debug', type=bool, default=False,
                        help='dumps the received and sent patches as png')
    

    args = parser.parse_args()


    # Checking provided context size
    if args.context_size not in (8, 16, 32, 64, 128):
        print("ERROR Unsupported context size %d" %(args.context_size))
        sys.exit(1)
        
    # Checking provided predictor size
    if args.predictor_size not in (4, 8, 16, 32, 64):
        print("ERROR Unsupported predictor size %d" %(args.predictor_size))
        sys.exit(1)
    
    # Checking provided pixel depth
    if args.bit_depth not in (8, 10):
        print("ERROR Unsupported pixel depth %d" %(args.bit_depth))
        sys.exit(1)
    
    # Loadinng the model
    if not isfile(args.model):
        print("ERROR Could not load model file %s" %(args.model))
        sys.exit(1)


    model = None
    if(args.model_type == 'mlp'):
        input_dim = ((args.predictor_size * args.n_lines)*2)+(args.n_lines*args.n_lines)
        hidden_dims = [
                20,
                10
            ]
            
        model = MLP(
            input_dim=input_dim,
            hidden_dims= hidden_dims,
            output_dim=1
        )

        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded checkpoint '{args.model}' (epoch {checkpoint['epoch']})")
    else:
        raise NotImplementedError(f'{args.model_type} not yet available')
    
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")

    print(model)
    # switch to evaluate mode
    model.eval()
    

    # Maximum size of the UDP messages we can receive over IP is 65507 (65,535 bytes max UDP payload − 8-byte UDP header − 20-byte IP header).
    expectedMessageSize = args.context_size * args.context_size * (2 if args.bit_depth == 10 else 1)

    # Binding the UDP socket to the desired port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((args.address, args.port))
    except OSError:
        print("ERROR Could not bind UDP socket at %s:%d, is another server already running ?" %(args.address, args.port))
        sys.exit(1)
    print("UDP server listening on %s port %d expecting messages of size %d bytes" %(args.address, args.port, expectedMessageSize))
    

    # Main cycle
    cnt = 0
    while True:
        # This is to cope with messages being fragmented into multiple UDP packets
        bytesToReceive = expectedMessageSize
        pktCounter = 0
        inDataBytes = b''
        while bytesToReceive > 0:
            # Watining to receive a message from the encoder
            recvDataBytes, address = sock.recvfrom(expectedMessageSize)
            # This is a debug request, not a context
            if (len(recvDataBytes) == 0 or len(recvDataBytes) == 1):
                break
            # Concatenatig byte objects here
            inDataBytes += recvDataBytes
            pktCounter = pktCounter + 1
            bytesToReceive = bytesToReceive - len(inDataBytes)
        
        s = time.time()
        print(str(cnt) + " received %s bytes in %d packets from %s " % (len(inDataBytes), pktCounter, address), end='')
        # inDataBytes will contain the pixels sent by the encoder, 8 or 10 bits per pixel, Y component only, [0 -> (2^bitDepth)-1] range
        
        # Here we send bach the commandline option as a form of remote debugging
        if (len(inDataBytes) == 0 or len(inDataBytes) == 1):
            sock.sendto(str(args).encode('utf-8'), address)
            continue
        
        # Converting uints to ushorts if bith depth > 0 (endianess left unaffected)
        #https://docs.python.org/3/library/struct.html
        #https://stackoverflow.com/questions/45187101/converting-bytearray-to-short-int-in-python
        if args.bit_depth > 8:
            inDataBytes = unpack('H'*(len(inDataBytes)//2), inDataBytes)
        
        # Reshaping the payload as a numpy array of float32 <class 'numpy.ndarray'> as required by the NN
        # inDataArray = np.zeros(len(inDataBytes))
        # for i in range(len(inDataBytes)):
        #    inDataArray[i] = int(inDataBytes[i])
        inDataArray =  np.array(inDataBytes, dtype=np.float32)
        
        if args.debug and args.bit_depth == 8:
            imageio.imwrite('./%d_rcvd.png' %(cnt), np.reshape(inDataArray.astype(np.uint8), (args.context_size, args.context_size)))
        # We stretch the dynamic 10->16 bit, otherwise the png will look all black
        if args.debug and args.bit_depth == 10:
            imageio.imwrite('./%d_rcvd.png' %(cnt), np.reshape(inDataArray.astype(np.uint16) * (2**6), (args.context_size, args.context_size)))
        
        # 1) reshape to get a square shape, e.g. 64x64 in NWHC format (1, 64, 64, 1)
        inputForNN = np.reshape(inDataArray, (1, args.context_size, args.context_size, 1))

        # 2) normalize inputForNN from  [0-<max pixel value>] to either [0,1] or [-1,1], depending how you trained your NN; eg inputForNNNorm = ((inputForNN/255) * 2) -1
        
        # TODO comment
        #predictor_evc = inputForNN[:,args.context_size-args.predictor_size:args.context_size, args.context_size-args.predictor_size:args.context_size,:]

        # TODO uncomment
        inputForNNNorm = inputForNN
        #inputForNNNorm = normalize(inputForNN, args.bit_depth)

        inputForNNNorm = torch.from_numpy(inputForNNNorm)
        inputForNNNorm = inputForNNNorm.reshape((1,1,args.context_size,args.context_size)) 

        # EVC mean + w
        #left=inputForNNNorm[:,:,args.context_size-args.predictor_size-1:,args.context_size-args.predictor_size-1:args.context_size-args.predictor_size]
        #top = inputForNNNorm[:,:,args.context_size-args.predictor_size-1:args.context_size-args.predictor_size,args.context_size-args.predictor_size:]

        #evc = torch.cat([top.reshape(-1),left.reshape(-1)],dim=0)
        #evc_prediction = (torch.sum(evc).item() + args.predictor_size ) / len(evc)
        #output = evc_prediction
        output = inputForNNNorm[:,:,-1,-1].item()

        # 3) feed inputForNN to the NN and get its output as, say, outputFromNN
        if(args.model_type == 'mlp' and False):
            # USE NET
            # takes the left parte of the hole
            #left=inputForNNNorm[:,:,args.context_size-args.predictor_size-args.n_lines:,args.context_size-args.predictor_size-args.n_lines:args.context_size-args.predictor_size]

            # takes the top part of the hole
            #top = inputForNNNorm[:,:,args.context_size-args.predictor_size-args.n_lines:args.context_size-args.predictor_size,args.context_size-args.predictor_size:]

            #x = torch.cat([top.reshape(-1),left.reshape(-1)],dim=0).unsqueeze(0)

            #print(x.shape)

            #x = x.to(device)
            #with torch.no_grad():
            #    output = model(x)
            
            # EVC Mean
            left=inputForNNNorm[:,:,args.context_size-args.predictor_size-1:,args.context_size-args.predictor_size-1:args.context_size-args.predictor_size]
            top = inputForNNNorm[:,:,args.context_size-args.predictor_size-1:args.context_size-args.predictor_size,args.context_size-args.predictor_size:]

            evc = torch.cat([top.reshape(-1),left.reshape(-1)],dim=0)
            evc_prediction = torch.mean(evc)
            output = evc_prediction 

        """ else:
            raise NotImplementedError(f'{args.model_type} not yet available') """


        # TODO uncomment
        
        outputFromNN = np.ones((1,args.predictor_size,args.predictor_size,1), dtype=np.float32) * output #.item()

        # TODO comment
        #outputFromNN = predictor_evc

        # 4) de-normalize the output towards [0-<max pixel value>], eg outputFromNNDenorm =  ((outputFromNN +1) /2) * 255
        
        # TODO uncomment
        outputFromNNDenorm = outputFromNN
        #outputFromNNDenorm = denormalize(outputFromNN, args.bit_depth)
        # And then clip into [0, (2**args.bit_depth) -1)], just au cas ou
        #outputFromNNDenorm = np.clip(np.around(outputFromNNDenorm), 0, (2**args.bit_depth) -1)

        # 5) cast the network output to uint8 or uint16
        if args.bit_depth == 8:
            outputFromNNDenorm  = outputFromNNDenorm.astype(np.uint8)
        else:
            outputFromNNDenorm  = outputFromNNDenorm.astype(np.uint16)
        
        e = time.time()

        print("time[ms] %d" % ((e-s)*1000))

        if args.debug and args.bit_depth == 8:
            imageio.imwrite('./%d_sent.png' %(cnt), np.reshape(outputFromNNDenorm, (args.predictor_size, args.predictor_size)))
        # We stretch the dynamic 10->16 bit, otherwise the png will look all black
        if args.debug and args.bit_depth == 10:
            imageio.imwrite('./%d_sent.png' %(cnt), np.reshape(outputFromNNDenorm  * (2**6), (args.predictor_size, args.predictor_size)))
        

        # Sending back the  to the encoder  a vector of uint8 or uint16
        sock.sendto(outputFromNNDenorm, address)
        cnt = cnt +1
