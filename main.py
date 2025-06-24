import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Main kNN-AS script")

    parser.add_argument("--k", type=int, default=5, 
            help="Number of neighbours")
    parser.add_argument("--n", type=int, default=6, 
            help="Number of states selected each round")
    parser.add_argument("--epochs", type=int, default=200, 
            help="Number of epochs")
    parser.add_argument("--n_steps", type=int, default=10000, 
            help="Number of time steps for each simulation")
    parser.add_argument("--trials", type=int, default=10, 
            help="Number of trials")
    parser.add_argument("--data_dir", type=str, default="", 
            help="Directory with starting points")
    parser.add_argument("--top_file", type=str, default="", 
            help="Topology file")
    parser.add_argument("--initial_structures", type=str, default="", 
            help="Initial starting points file name")
    parser.add_argument("--restart", type=bool, default=False, 
            help="Restart")
    parser.add_argument("--algorithm", type=str, default="", 
            help="Whether algorithm 1 (sum of vectors) or algorithm 2 (average distance)")


    return parser.parse_args()

def main(args):
    k=5
    n=6
    epochs = 40 
    n_steps = 10000000
    trials = 2
    cmap = plt.get_cmap(name='hsv')
    
    main_dir = os.getcwd()
    data_dir = main_dir+'/example_data/'
    
    top_file = data_dir+'4AKE_water.pdb'
    initial_structures = [data_dir+'npt_1AKE.pdb',data_dir+'npt_4AKE.pdb',
                          data_dir+'npt_1AKE.pdb',data_dir+'npt_4AKE.pdb',
                          data_dir+'npt_1AKE.pdb',data_dir+'npt_4AKE.pdb']

    main_dir = os.getcwd()

    for t in range(1,trials):
        directory = dirname+f"_{t}"
        path = os.path.join(main_dir, directory)
        if os.path.isdir(path) & !restart:
            FileExistsError(f"Directory already exists: {dir_path}, did you mean restart?")
        elif !os.path.isdir(path) & !restart:
            os.makedirs(path)
        os.chdir(path)

        if restart:
            file = glob('features_tmp*.npy')[0]
            angle_total = list(np.load(file))
            idx_pos =  list(np.load('idx_tmp.npy')
            a,b,c = np.array(angle_total).shape
            X_tmp = np.array(angle_total).reshape(a*b,c)
            idx = np.random.choice(len(X_tmp), int(len(X_tmp)*0.2), replace=False)
            X_tmp2 = X_tmp[idx]
            selection = KNN_sampling(X_tmp2, k, n)
            selection = idx[selection]
            a,b,c = np.array(idx_pos).shape
            idx_tmp = np.array(idx_pos).reshape(a*b,c)
            sys.stdout.write(f"Finished KNN")
            sys.stdout.flush()
            print(idx_tmp)
        else:


        initial_positions = []
        sys.stdout.write(f"Start loading MD")
        sys.stdout.flush()
        for sel in selection:
            o_f,o = idx_tmp[sel]
            traj = md.load_frame(o_f, int(o), top=top_file)
            initial_positions.append(np.squeeze(traj.xyz, axis=0))
            del traj
        
        print(initial_positions)        
        num=0
        num2 = 0
        for i in range(15, epochs+1): # number of epochs
            sys.stdout.write(f"Epoch: {i}")
            sys.stdout.flush()
            
            procs = []
            for j in range(n):
                print(f"start: {j}")
                adk_ = ADK(i, j, initial_positions, top_file, n_steps)
                with open(f"simulation{j}.pkl", "wb") as f:
                    pickle.dump(adk_, f)
                cmd = ["srun",
                    "--het-group=1",
                    "--ntasks=1",
                    "--gpus-per-task=1",
                    "--cpus-per-task=1",         # Optional: 1 CPU per task
                    "python",
                    main_dir+"/gpu_worker.py",
                    f"simulation{j}.pkl"
                    ]
                sys.stdout.write(f"Launching task {j}\n")
                procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

            sys.stdout.write(f"Finished pickle")
            sys.stdout.flush()

            for proc in procs:
                stdout, stderr = proc.communicate()
                if proc.returncode == 0:
                    output = stdout.decode().strip()
                    output_file = re.findall(f'results.*\.pkl', output)[0]
                    with open(output_file, 'rb') as f:
                        results = pickle.load(f)
                    angle_total.append(results[0])
                    o_file = results[1]
                    tmp_r = [[o_file,i] for i in range(len(results[0]))]
                    idx_pos.append(tmp_r)
                else:
                    print(f"Worker error:\n{stderr.decode()}")
                    raise
            print(idx_pos)            
            for f in glob.glob("*.pkl"):
                os.remove(f)

            #calculate new starting positions
            sys.stdout.write(f"Start KNN")
            sys.stdout.flush()

            a,b,c = np.array(angle_total).shape
            X_tmp = np.array(angle_total).reshape(a*b,c)
            idx = np.random.choice(len(X_tmp), int(len(X_tmp)*0.2), replace=False)
            X_tmp2 = X_tmp[idx]
            selection = KNN_sampling(X_tmp2, k, n)
            selection = idx[selection]
            a,b,c = np.array(idx_pos).shape
            idx_tmp = np.array(idx_pos).reshape(a*b,c)
            sys.stdout.write(f"Finished KNN")
            sys.stdout.flush()
            print(idx_tmp)
            initial_positions = []
            sys.stdout.write(f"Start loading MD")
            sys.stdout.flush()
            for sel in selection:
                o_f,o = idx_tmp[sel]
                traj = md.load_frame(o_f, int(o), top=top_file)
                initial_positions.append(np.squeeze(traj.xyz, axis=0))
                del traj
            sys.stdout.write(f"Finished loading MD")
            sys.stdout.flush()
            os.remove('angels_tmp*')
            np.save(f'angels_tmp_{i}.npy', X_tmp)

        np.save('angels_final.npy', X_tmp)
        os.chdir(main_dir)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

