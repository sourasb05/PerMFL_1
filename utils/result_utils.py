import h5py
import numpy as np
import pandas as pd 
import os
import numpy as np
import re
import csv
def convert_csv_to_txt(input_file,output_file):
   
    with open(input_file, 'r') as csv_file, open(output_file, 'w') as space_delimited_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            space_delimited_file.write(' '.join(row) + '\n')

    print(f'CSV file "{input_file}" converted to space-delimited file "{output_file}"')



def read_file(file):
    hf = h5py.File(file, 'r')
    attributes = []
    for key in hf.keys():
        attributes.append(key)
    
    return attributes, hf


def get_data(hf,attributes):
    data = []
    pm = []
    acc_pm = []
    loss_pm = []
    loss_gm = []
    for i in range(len(attributes)):
        ai = hf.get(attributes[i])
        ai = np.array(ai)
        data.append(ai)
    
    return data


def convergence_analysis(path, acc_file, loss_file):
    dir_list = os.listdir(path)
    
    permfl_per_test_loss = []
    permfl_per_test_accuracy = []
    permfl_global_test_loss = []
    permfl_global_test_accuracy = []

    AL2GD_per_test_loss = []
    AL2GD_per_test_accuracy = []
    hsgd_global_test_loss = []
    hsgd_global_test_accuracy = []

    client_gen_acc = []
    client_spe_acc = []
    client_spe_loss = []
    client_gen_loss = []
    for file_name in dir_list:
        if file_name in ['permfl.h5','hsgd.h5', 'AL2GD.h5']:
            print(file_name)
            attributes, hf = read_file(path+file_name)

            data = get_data(hf,attributes)
            id=0
            for key in hf.keys():
                attributes.append(key)
                # print("id [",id,"] :", key)
                id+=1
                
            ptrl = hf.get('per_train_loss')
            ptra = hf.get('per_train_accuracy')
            ptsl = hf.get('per_test_loss')
            ptsa = hf.get('per_test_accuracy')

            gtrl = hf.get('global_train_loss')
            gtra = hf.get('global_train_accuracy')   
            gtsl = hf.get('global_test_loss')
            gtsa = hf.get('global_test_accuracy')
                
            if file_name == "permfl.h5":
                # for perMFL cnn trl is the test accuracy
                permfl_per_test_loss.append(np.array(ptsl).tolist())
                permfl_per_test_accuracy.append(np.array(ptsa).tolist())

                permfl_global_test_loss.append(np.array(gtsl).tolist())
                permfl_global_test_accuracy.append(np.array(gtra).tolist())

                print("permfl_per_test_loss :",len(permfl_per_test_loss))
                print("permfl_per_test_acccuracy :",len(permfl_per_test_accuracy))
                print("permfl_global_test_loss :",len(permfl_global_test_loss))
                print("permfl_global_test_accuracy :",len(permfl_global_test_accuracy))

            if file_name == "hsgd.h5":
                hsgd_global_test_loss.append(np.array(gtsl).tolist())
                hsgd_global_test_accuracy.append(np.array(gtsa).tolist())


                # print("hsgd_test_accuracy :",hsgd_global_test_accuracy)
                # print("hsgd_test_loss :", hsgd_global_test_loss)
            
            if file_name == "AL2GD.h5":
                AL2GD_per_test_loss.append(np.array(ptsl).tolist())
                AL2GD_per_test_accuracy.append(np.array(ptsa).tolist())

            #  print("AL2GD_per_test_accuracy :",AL2GD_per_test_accuracy)
                # print("AL2GD_per_test_loss :", AL2GD_per_test_loss)
        elif file_name == 'demlearn.h5':  
            print(file_name)
            attributes, hf = read_file(path+file_name)

            data = get_data(hf,attributes)
            id=0
            for key in hf.keys():
                attributes.append(key)
                print("id [",id,"] :", key)
                id+=1
            
            cgen_acc = hf.get('cg_avg_data_test')
            cspe_acc = hf.get('cs_avg_data_test')
            cgen_loss = hf.get('cg_loss_test')
            cspe_loss = hf.get('cs_loss_test')


            # print(cgen_acc)
            # input("press")

            client_gen_acc.append(np.array(cgen_acc).tolist())
            client_spe_acc.append(np.array(cspe_acc).tolist())
            client_gen_loss.append(np.array(cgen_loss).tolist())
            client_spe_loss.append(np.array(cspe_loss).tolist())

           # print(client_gen_acc)
           # print(client_spe_acc)
           # input("press")
            

    data_loss = {
        'GR' : np.arange(100),
        'PerMFL(PM)' :  permfl_per_test_loss[0][:100],
        'PerMFL(GM)' :  permfl_global_test_loss[0][:100],
        'h-sgd' :   hsgd_global_test_loss[0][:100],
        'AL2GD' : AL2GD_per_test_loss[0][:100],
        'DemLearn(C-GEN)' : client_gen_loss[0][:100],
        'DemLearn(C-SPE)' : client_spe_loss[0][:100] 
             
                    }

    data_acc = {
        'GR'    : np.arange(100),
        'PerMFL(PM)' :  permfl_per_test_accuracy[0][:100],
        'PerMFL(GM)' :  permfl_global_test_accuracy[0][:100],
        'h-sgd' :   hsgd_global_test_accuracy[0][:100],
        'AL2GD' : AL2GD_per_test_accuracy[0][:100],
        'DemLearn(C-GEN)' : client_gen_acc[0][:100],
        'DemLearn(C-SPE)' : client_spe_acc[0][:100]     
                    }
    df_l = pd.DataFrame(data_loss)
    df_a = pd.DataFrame(data_acc)

    csv_acc_path = path + acc_file +".csv"
    csv_loss_path = path + loss_file +".csv"
    txt_acc_path = path + acc_file +".txt"
    txt_loss_path = path + loss_file +".txt"

    df_l.to_csv(csv_loss_path, index=False)
    df_a.to_csv(csv_acc_path, index=False)

    convert_csv_to_txt(csv_acc_path,txt_acc_path)
    convert_csv_to_txt(csv_loss_path,txt_loss_path)
    

    print(df_l)
    print(df_a)


    




def AL2GD_plots(path):
    AL2GD_per_test_loss = []
    AL2GD_per_test_accuracy = []

    dir_list = os.listdir(path)

    for file_name in dir_list:
        
        print(file_name)
        attributes, hf = read_file(path+file_name)

        data = get_data(hf,attributes)
        id=0
        for key in hf.keys():
            attributes.append(key)
            # print("id [",id,"] :", key)
            id+=1
            
        # ptrl = hf.get('per_train_loss')
        # ptra = hf.get('per_train_accuracy')
        ptsl = hf.get('per_test_loss')
        ptsa = hf.get('per_test_accuracy')

        AL2GD_per_test_loss.append(np.array(ptsl).tolist())
        AL2GD_per_test_accuracy.append(np.array(ptsa).tolist())
    
    data_loss = {
        'AL2GD' : AL2GD_per_test_loss[0][:100:10]      
                    }

    data_acc = {
        'AL2GD' : AL2GD_per_test_accuracy[0][:100:10]      
                    }
    df_l = pd.DataFrame(data_loss)
    df_a = pd.DataFrame(data_acc)

    df_l.to_csv(path + 'perf_loss_AL2GD.csv', index=False)
    df_a.to_csv(path + 'perf_acc_AL2GD.csv', index=False)

    print(df_l)
    print(df_a)




def average_result(path, algorithm, ul):
    
    dir_list = os.listdir(path)

   
    
    gtr_loss = []
    gts_loss = []
    gtr_acc = []
    gts_acc = []
    
        
    i=0
    
    if algorithm in ["PerMFL", "AL2GD", "pFedMe","ditto"] :

        ptr_loss = []
        pts_loss = []
        ptr_acc = []
        pts_acc = []

        for file_name in dir_list:
            if file_name.endswith(".h5"):
                print(file_name)
                attributes, hf = read_file(path+file_name)

                data = get_data(hf,attributes)
                id=0
                for key in hf.keys():
                    attributes.append(key)
                    # print("id [",id,"] :", key)
                    id+=1
            
                ptsl = hf.get('per_test_loss')
                ptsa = hf.get('per_test_accuracy')
                ptrl = hf.get('per_train_loss')
                ptra = hf.get('per_train_accuracy')
            
                pts_loss.append(np.min(ptsl[:ul]))
                ptr_loss.append(np.min(ptrl[:ul]))
                pts_acc.append(np.max(ptsa[:ul]))
                ptr_acc.append(np.max(ptra[:ul]))
            
                gtsl = hf.get('global_test_loss')
                gtrl = hf.get('global_train_loss')
                gtsa = hf.get('global_test_accuracy')
                gtra = hf.get('global_train_accuracy')

                gts_loss.append(np.min(gtsl[:ul]))
                gtr_loss.append(np.min(gtrl[:ul]))
                gts_acc.append(np.max(gtsa[:ul]))
                gtr_acc.append(np.max(gtra[:ul]))
            
        
            ptrl_mean = np.array(ptr_loss).mean()
            ptsl_mean = np.array(pts_loss).mean()
            ptra_mean = np.array(ptr_acc).mean()
            ptsa_mean = np.array(pts_acc).mean()

            ptrl_std = np.array(ptr_loss).std()
            ptsl_std = np.array(pts_loss).std()
            ptra_std = np.array(ptr_acc).std()
            ptsa_std = np.array(pts_acc).std()
            
            gtrl_mean = np.array(gtr_loss).mean()
            gtsl_mean = np.array(gts_loss).mean()
            gtra_mean = np.array(gtr_acc).mean()
            gtsa_mean = np.array(gts_acc).mean()

            gtrl_std = np.array(gtr_loss).std()
            gtsl_std = np.array(gts_loss).std()
            gtra_std = np.array(gtr_acc).std()
            gtsa_std = np.array(gts_acc).std()

            # print("personalized training loss (mean/std) : (",ptrl_mean,"/",ptrl_std,")")
            # print("personalized test loss (mean/std) : (",ptsl_mean,"/",ptsl_std,")")
            print("personalized training accuracy (mean/std) : (",ptra_mean,"/",ptra_std,")")
            print("personalized test accuracy (mean/std) : (",ptsa_mean,"/",ptsa_std,")")

            # print("Global training loss (mean/std) : (",gtrl_mean,"/",gtrl_std,")")
            # print("Global test loss (mean/std) : (",gtsl_mean,"/",gtsl_std,")")
            print("Global training accuracy (mean/std) : (",gtra_mean,"/",gtra_std,")")
            print("Global test accuracy (mean/std) : (",gtsa_mean,"/",gtsa_std,")")
    
    elif algorithm in [ "FedAvg", "h-sgd", "Hier-LQSGD" ]:
        for file_name in dir_list:
            if file_name.endswith(".h5"):
                print(file_name)
                attributes, hf = read_file(path+file_name)

                data = get_data(hf,attributes)
                id=0
                for key in hf.keys():
                    attributes.append(key)
                    # print("id [",id,"] :", key)
                    id+=1

                gtsl = hf.get('global_test_loss')
                gtrl = hf.get('global_train_loss')
                gtsa = hf.get('global_test_accuracy')
                gtra = hf.get('global_train_accuracy')

                gts_loss.append(np.min(gtsl[:ul]))
                gtr_loss.append(np.min(gtrl[:ul]))
                gts_acc.append(np.max(gtsa[:ul]))
                gtr_acc.append(np.max(gtra[:ul]))
        
        gtrl_mean = np.array(gtr_loss).mean()
        gtsl_mean = np.array(gts_loss).mean()
        gtra_mean = np.array(gtr_acc).mean()
        gtsa_mean = np.array(gts_acc).mean()

        gtrl_std = np.array(gtr_loss).std()
        gtsl_std = np.array(gts_loss).std()
        gtra_std = np.array(gtr_acc).std()
        gtsa_std = np.array(gts_acc).std()

        print("Global training loss (mean/std) : (",gtrl_mean,"/",gtrl_std,")")
        print("Global test loss (mean/std) : (",gtsl_mean,"/",gtsl_std,")")
        print("Global training accuracy (mean/std) : (",gtra_mean,"/",gtra_std,")")
        print("Global test accuracy (mean/std) : (",gtsa_mean,"/",gtsa_std,")")

    elif algorithm == "PerAvg":
        ptr_loss = []
        pts_loss = []
        ptr_acc = []
        pts_acc = []

        for file_name in dir_list:
            print(file_name)
            attributes, hf = read_file(path+file_name)

            data = get_data(hf,attributes)
            id=0
            for key in hf.keys():
                attributes.append(key)
                print("id [",id,"] :", key)
                id+=1

            ptsl = hf.get('per_test_loss')
            ptrl = hf.get('per_train_loss')
            ptsa = hf.get('per_test_accuracy')
            ptra = hf.get('per_train_accuracy')

            pts_loss.append(np.min(ptsl[:]))
            ptr_loss.append(np.min(ptrl[:]))
            pts_acc.append(np.max(ptsa[:]))
            ptr_acc.append(np.max(ptra[:]))
        
        ptrl_mean = np.array(ptr_loss).mean()
        ptsl_mean = np.array(pts_loss).mean()
        ptra_mean = np.array(ptr_acc).mean()
        ptsa_mean = np.array(pts_acc).mean()

        ptrl_std = np.array(ptr_loss).std()
        ptsl_std = np.array(pts_loss).std()
        ptra_std = np.array(ptr_acc).std()
        ptsa_std = np.array(pts_acc).std()

        print("Global training loss (mean/std) : (",ptrl_mean,"/",ptrl_std,")")
        print("Global test loss (mean/std) : (",ptsl_mean,"/",ptsl_std,")")
        print("Global training accuracy (mean/std) : (",ptra_mean,"/",ptra_std,")")
        print("Global test accuracy (mean/std) : (",ptsa_mean,"/",ptsa_std,")")





def convergence_analysis_PerMFL(path):
    dir_list = os.listdir(path)

    permfl_per_test_loss = []
    permfl_per_test_accuracy = []
    permfl_global_test_loss = []
    permfl_global_test_accuracy = []

    column_name = []

    for file_name in dir_list:
        
        print(file_name)

        match = re.search(r'TE_\d+', file_name)

        if match:
            round_number = match.group()
            print(round_number)  # Output: round4
            column_name.append(round_number)
        else:
            print("No round number found.")
        
        attributes, hf = read_file(path+file_name)

        data = get_data(hf,attributes)
        id=0
        for key in hf.keys():
            attributes.append(key)
            # print("id [",id,"] :", key)
            id+=1
            
        ptrl = hf.get('per_train_loss')
        ptra = hf.get('per_train_accuracy')
        ptsl = hf.get('per_test_loss')
        ptsa = hf.get('per_test_accuracy')

        gtrl = hf.get('global_train_loss')
        gtra = hf.get('global_train_accuracy')   
        gtsl = hf.get('global_test_loss')
        gtsa = hf.get('global_test_accuracy')
            
        permfl_per_test_loss.append(np.array(ptsl).tolist())
        permfl_per_test_accuracy.append(np.array(ptsa).tolist())

        permfl_global_test_loss.append(np.array(gtsl).tolist())
        permfl_global_test_accuracy.append(np.array(gtra).tolist())


    data_loss = {
        'p'+column_name[0] : permfl_per_test_loss[0][:500:20],
        'p'+column_name[1] : permfl_per_test_loss[1][:500:20],      
        'g'+column_name[0] : permfl_global_test_loss[0][:500:20],
        'g'+column_name[1] : permfl_global_test_loss[1][:500:20]      
                }

    data_acc = {
        'p'+column_name[0] : permfl_per_test_accuracy[0][:500:20],
        'p'+column_name[1] : permfl_per_test_accuracy[1][:500:20],  
        'g'+column_name[0] : permfl_global_test_accuracy[0][:500:20],
        'g'+column_name[1] : permfl_global_test_accuracy[1][:500:20]
                    }
        
    df_l = pd.DataFrame(data_loss)
    df_a = pd.DataFrame(data_acc)

    df_l.to_csv('./results/' + 'PerMFL_perf_loss_ptfc.csv', index=False)
    df_a.to_csv('./results/' + 'PerMFL_perf_acc_ptfc.csv', index=False)

    print(df_l)
    print(df_a)

    import csv

# Input CSV file and output space-delimited file
    input_file = './results/' + 'PerMFL_perf_loss_ptfc.csv'
    output_file = './results/' + 'PerMFL_perf_loss_ptfc.txt'

    with open(input_file, 'r') as csv_file, open(output_file, 'w') as space_delimited_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            space_delimited_file.write(' '.join(row) + '\n')

    print(f'CSV file "{input_file}" converted to space-delimited file "{output_file}"')



def PerMFL_ablation_plot(path):

    dir_list = os.listdir(path)

    permfl_per_test_loss = []
    permfl_per_test_accuracy = []
    permfl_global_test_loss = []
    permfl_global_test_accuracy = []

    column_name = []
    for i, file_name in enumerate(dir_list):
        
        print(file_name)
        if file_name.lower().endswith('.h5') == 1:
        #    match = re.search(r'lamdba_0_\d+(\.\d+)?', file_name)
        #    print(match)

        #    if match:
        #        matched_text = match.group()
        #        print(matched_text)  # Output: round4
        #        column_name.append(matched_text)
            #else:
            #    print("No round number found.")
            
            column_name.append(file_name[0:11])
            attributes, hf = read_file(path+file_name)

            data = get_data(hf,attributes)
            id=0
            for key in hf.keys():
                attributes.append(key)
                # print("id [",id,"] :", key)
                id+=1
                
            ptrl = hf.get('per_train_loss')
            ptra = hf.get('per_train_accuracy')
            ptsl = hf.get('per_test_loss')
            ptsa = hf.get('per_test_accuracy')

            gtrl = hf.get('global_train_loss')
            gtra = hf.get('global_train_accuracy')   
            gtsl = hf.get('global_test_loss')
            gtsa = hf.get('global_test_accuracy')
                
            permfl_per_test_loss.append(np.array(ptsl).tolist())
            permfl_per_test_accuracy.append(np.array(ptsa).tolist())

            permfl_global_test_loss.append(np.array(gtsl).tolist())
            permfl_global_test_accuracy.append(np.array(gtra).tolist())

            #print("permfl_per_test_loss : \n",len(permfl_per_test_loss[i]))
            #print("permfl_per_test_accuracy : \n",len(permfl_per_test_accuracy[i]))
            #print("permfl_per_test_accuracy : \n", len(permfl_global_test_loss[i]))
            #print("permfl_per_test_accuracy : \n",len(permfl_global_test_accuracy[i]))
    
    data_loss = {
        'p'+ column_name[0] : permfl_per_test_loss[0][:100:10],
        'p'+ column_name[1] : permfl_per_test_loss[1][:100:10],      
         'p'+ column_name[2] : permfl_per_test_loss[2][:100:10],
        # 'p'+ column_name[3] : permfl_per_test_loss[3][:100:10],
        'g'+ column_name[0] : permfl_global_test_loss[0][:100:10],
        'g'+ column_name[1] : permfl_global_test_loss[1][:100:10],      
         'g'+ column_name[2] : permfl_global_test_loss[2][:100:10],
       # 'g'+ column_name[3] : permfl_global_test_loss[3][:100:10]     
        
        }

    data_acc = {
        'p'+column_name[0] : permfl_per_test_accuracy[0][:100:10],
        'p'+column_name[1] : permfl_per_test_accuracy[1][:100:10], 
        'p'+column_name[2] : permfl_per_test_accuracy[2][:100:10],
        # 'p'+column_name[3] : permfl_per_test_accuracy[3][:100:10],
        'g'+column_name[0] : permfl_global_test_accuracy[0][:100:10],
        'g'+column_name[1] : permfl_global_test_accuracy[1][:100:10],
        'g'+column_name[2] : permfl_global_test_accuracy[2][:100:10],
       # 'g'+column_name[3] : permfl_global_test_accuracy[3][:100:10]
        
    }

    df_l = pd.DataFrame(data_loss)
    df_a = pd.DataFrame(data_acc)
    

    df_l.to_csv(path + 'PerMFL_part_team_part_client_loss.csv', index=False)
    df_a.to_csv(path + 'PerMFL_part_team_part_client_acc.csv', index=False)

    print("df_l : \n",df_l)
    print("df_a : \n",df_a)




