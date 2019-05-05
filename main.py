import argparse
import sys



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='There are three mode:\
            \"player\" for human player,\
            \"train\" for training the AI player,\
            \"ai\" for ai player mode.')
    parser.add_argument('--mode',default="train", type=str)

    args=parser.parse_args()

    if args.mode == "train":
        sys.path.append("./train_2048/")
        import main 
        main.main()
        print("train")
    elif args.mode == "player":
        sys.path.append("./player_2048/")
        import puzzle 
        puzzle.main()
        print("player")
    elif args.mode == "ai":
        sys.path.append("./ai_2048/")
        import main 
        main.main()
        print("ai")
    else:
        print("error!")
