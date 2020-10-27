import os 
import argparse

if __name__ == '__main__':
    if(os.path.isdir('./Results')):
        os.system('rm -rf Results')
        print("Deleted Results dir")
        #os.system('mkdir Results')
    #os.makedirs('Results', mode=0o777)
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="example_videos")
    args=vars(ap.parse_args())
    command="python3 video_another_emotion.py -v "+str(args["video"])
    print(command)
    os.system(command)
    command2="python3 cluster_chinese.py -d Results/Extracted_frames/Extracted_faces/"
    print(command2)
    os.system(command2)
    os.system("cp face_emotion_data_analysis.py ./Results/")
    print("Clustering Done")
    os.system("python3 ./Results/face_emotion_data_analysis.py")

    os.system("python3 fusion.py")
    
 


