void reverse(char* bytes, int numChunks){
    for(int i =0 ; i<numChunks; ++i){
        for(int j=0; j<32; ++j){
            char tmp = bytes[i*64+j];
            bytes[i*64+j]= bytes[(i+1)*64-j-1];
            bytes[(i+1)*64-j-1] = tmp;
        }
    }
}