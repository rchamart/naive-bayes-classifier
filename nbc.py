import csv
import pandas as pd
import numpy as np
import distutils
import os
import sys

module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)


class NaiveBaes:
    def imp(self, data_path):
        # IMPORT THE DATA
        return pd.read_csv(data_path)

    def zeroAndSquares(self, trained_data, actual):
        #print "ACTUAL DATAVALUES: "
        #print actual
        #print
        #print "TRAINED DATAVALUES: "
        #print trained_data

        actual_values = actual.tolist()
        #print "NUMBER OF VALUES THAT EXIST: "
        total_n = len(actual_values)
        #print (total_n)

        # MERGE INTO A DATAFRAME FOR ITERATION: ISSA VIBE
        a_merged_table = pd.DataFrame({'TRAINED': trained_data, 'ACTUAL': actual_values})

        # MAP BINARY --> INT
        a_new_tvalue = a_merged_table["TRAINED"].to_list()
        a_new_tvalue = np.asarray(a_new_tvalue)
        a_new_tvalue = a_new_tvalue.astype(int)

        a_new_actvalue = a_merged_table["ACTUAL"].to_list()
        a_new_actvalue = np.asarray(a_new_actvalue)
        a_new_actvalue = a_new_actvalue.astype(int)

        # GET NUMS OF 0S AND 1S REQ FOR ZERO-ONE LOSS
        j = 0
        for r in range(0, total_n):
            if a_new_actvalue[r] != a_new_tvalue[r]:
                j += 1

        # MAP BINARY --> INT
        a_new_tvalue = a_merged_table["TRAINED"].to_list()
        a_new_tvalue = np.asarray(a_new_tvalue)
        a_new_tvalue = a_new_tvalue.astype(int)

        a_new_actvalue = a_merged_table["ACTUAL"].to_list()
        a_new_actvalue = np.asarray(a_new_actvalue)
        a_new_actvalue = a_new_actvalue.astype(int)

        # CALCULATE MSE(MEAN SQUARED ERROR)
        # print("MSE: ")
        constant_sum = 0
        for i in range(0, total_n):
            sq_diff = ((a_new_actvalue[i] - a_new_tvalue[i])**2)
            constant_sum += sq_diff
        mse = (1-(constant_sum/(total_n*1.0)))
        # print(mse)

        # CALCULATE THE LOSS VALUE
        # print("ZERO-ONE LOSS VALUE: ")
        loss_val = j * (1.0/total_n)
        # print(loss_val)

        # return and mse and loss values to the boyos
        estimate_array = [mse, loss_val]
        return estimate_array






    def calcNBC(self, hash, train_data):
        print train_data

        #print hash.get("p_lotT")["True"]

        # CREATE AN ARRAY TO STORE OUR DATA SO WE CAN CHECK IT OUT IN OUR EVALUATION FUNCTIONS
        results = []

        # GO ROW BY ROW AND PREDICT!!!!!!!
        for index, r in train_data.iterrows():
            # GET ROW VALUE AND SETUP REQUIRED PARAMS
            row = train_data.loc[[index]]
            #print row
            nbc_false = 1.0
            nbc_true = 1.0

            # PARSE AND GET THE VALUES FOR ALLL THE HOMIES IN THE ROW
            for col in row:
                # GET THE FEATURE HASH ACCESS SETUP
                true = col + 'T'
                false = col + 'F'

                # GET VALUE IN THAT PARTICULAR FEATURE COLUMN
                state = pd.Series.to_string(row[col], index=False)
                state = state[1:]
                #print state

                state_tval = 1.0
                state_fval = 1.0

                # FORMAT PARSED VALUES FOR ACCESS
                if col == "latitude" or col == "longitude":
                    if state != "None":
                        state = int(state)

                elif col == "stars" or col == "priceRange":
                    if state != "None":
                        state = float(state)

                elif col != "state" or col != "alcohol" or col != "noiseLevel" or col != "attire" or col != "smoking":
                    if state == "True":
                        state = True
                    elif state == "False":
                        state = False

                # ACCESS THE HASH FOR GIVEN VALUES
                try:
                    state_tval = hash.get(true)[state]
                    state_fval = hash.get(false)[state]
                except KeyError as e:
                    state_tval = 1
                    state_fval = 1
                except IndexError as f:
                    # print 'I got an IndexError - reason "%s"' % str(f)
                    state_tval = 1
                    state_fval = 1
                except TypeError as g:
                    # print 'I got an TypeError - reason "%s"' % str(g)
                    state_fval = 1
                    state_tval = 1

                # CALCULATE NBC VALUES
                # print true
                # print state_tval
                nbc_false = nbc_false*state_fval

                # print false
                # print state_fval
                nbc_true = nbc_true*state_tval

            # MULTIPLY THE CLASS PRIOR IN
            nbc_false = float(hash.get("prior")[False] * nbc_false)
            #print nbc_false

            nbc_true = float(hash.get("prior")[True] * nbc_true)
            #print nbc_true

            # FIND THE MAXIMUM BETWEEN NBC_FALSE AND NBC_TRUE
            nbc_final = max(nbc_true, nbc_false)
            value = None
            if nbc_true == nbc_final:
                # print ("TRUE TRUE TRUE")
                value = True
            else:
                # print ("FALSE FALSE FALSE")
                value = False

            # PRINTING THE NBC VALUE HERE BUT THIS IS WHERE WE WILL ADD IT TO OUR NBC ARRAY
            # print nbc_final
            results.append(value)

        return results


    def buildMLE(self, data):
        # DICTIONARY DEFINITION
        hash = {}

        # CLASS PRIOR
        # CALCULATE:
        # print("CLASS PRIOR: P(OUTDOOR_SEATING)")
        classify_num_values = data['outdoorSeating'].value_counts()
        d_false = classify_num_values[0]
        d_true = classify_num_values[1]

        total_aka_denom = classify_num_values[0] + classify_num_values[1]
        classify_tots = classify_num_values / total_aka_denom

        # STORE:
        hash["prior"] = classify_tots
        # print (classify_tots)

        # print
        # print

        ''' CALCULATE THE CPDS FOR ALL PARAMS INDEPENDENTLY ALA NBC AND STORE IN DICTIONARY
        
            FORMAT FOR ACCESS: 
            P(feature | Y=(T | F)) = hash.get("feature_name + (T|F)")
            
            CLASS PRIOR = hash.get("prior")'''

        for cols in data.columns:
            # FIRST FIND THE PROBABILITY GIVEN OUTDOOR_SEATING=FALSE
            #print "P(%s | OUTDOOR_SEATING=FALSE)"%cols
            # CALCULATE:
            t_false = data[data['outdoorSeating'] == False]
            t_fnum = t_false.groupby([cols, 'outdoorSeating']).size()

            # STORE IN THE HASHMAP
            try:
                # convert the number of Nones to an int
                numNone = int(t_fnum["None"])

                # drop the number of Nones from the denom
                d_new = d_false - numNone

                # drop the Nones
                t_exp = t_fnum.drop(['None'])

                # divide to create new shit
                new_experiment = t_exp / d_new

                # add into the hash
                keyF = cols + 'F'
                hash[keyF] = new_experiment
            except KeyError as e:
                # SOME ERROR HANDLING
                # print 'I got a KeyError - "%s" DOES NOT EXIST' % str(e)

                # CALCULATE
                t_divf_num = t_fnum / d_false

                # STORE:
                keyF = cols + 'F'
                hash[keyF] = t_divf_num

            # FIRST FIND THE PROBABILITY GIVEN OUTDOOR_SEATING=TRUE
            # print "P(%s | OUTDOOR_SEATING=TRUE)"%cols
            # CALCULATE:
            t_true = data[data['outdoorSeating'] == True]
            t_tnum = t_true.groupby([cols, 'outdoorSeating']).size()

            # STORE IN THE HASHMAP
            try:
                # convert the number of Nones to an int
                numNone2 = int(t_tnum["None"])

                # drop the number of Nones from the denom
                d_n = d_true - numNone2

                # drop the Nones
                t_exp = t_tnum.drop(['None'])

                # divide to create new shit
                newt_experiment = t_exp / d_n

                # add into the hash
                keyT = cols + 'T'
                hash[keyT] = newt_experiment
            except KeyError as e:
                # SOME ERROR HANDLING JUST FOR ME
                # print 'I got a KeyError - "%s" DOES NOT EXIST' % str(e)

                # CALCULATE
                t_divt_num = t_tnum / d_true

                # STORE:
                keyT = cols + 'T'
                hash[keyT] = t_divt_num

        # RETURN A HASHMAP CONTAINING ALL OF OUR INDIVIDUALLY CALCULATED MLE VIA NBC
        return hash

    def preprocess(self, data):
        # EXPAND AMBIANCE
        # AMBIANCE HAS 9 POSSIBLE VALUES
        # * romantic
        # * intimate
        # * classy
        # * hipster
        # * trendy
        # * casual
        # * divey
        # * upscale
        # * touristy
        r3 = pd.Series(data['ambience'])
        r3 = r3.fillna("None")

        # romantic
        a_rom = r3.str.contains('romantic', regex=False)
        amb_romantic = a_rom.fillna(False)

        # intimate
        a_int = r3.str.contains('intimate', regex=False)
        amb_intimate = a_int.fillna(False)

        # classy
        a_class = r3.str.contains('classy', regex=False)
        amb_classy = a_class.fillna(False)

        # hipster
        a_hips = r3.str.contains('hipster', regex=False)
        amb_hipster = a_hips.fillna(False)

        # trendy
        a_trendy = r3.str.contains('trendy', regex=False)
        amb_trendy = a_trendy.fillna(False)

        # casual
        a_casual = r3.str.contains('casual', regex=False)
        amb_casual = a_casual.fillna(False)

        # divey
        a_dive = r3.str.contains('divey', regex=False)
        amb_divey = a_dive.fillna(False)

        # upscale
        a_ups = r3.str.contains('upscale', regex=False)
        amb_upscale = a_ups.fillna(False)

        # touristy
        a_tourist = r3.str.contains('touristy', regex=False)
        amb_touristy = a_tourist.fillna(False)



        # EXPAND PARKING
        '''
            PARKING HAS 5 POSSIBLE VALUES
            * garage 
            * street 
            * validate 
            * lot 
            * valet 
        '''
        #setup
        r40 = pd.Series(data['parking'])
        r40 = r40.fillna("None")

        #garage
        p_garage = r40.str.contains('garage', regex=False)
        p_garage = p_garage.fillna(False)

        #street
        p_street = r40.str.contains('street', regex=False)
        p_street = p_street.fillna(False)

        #validate
        p_validate = r40.str.contains('validate', regex=False)
        p_validate = p_validate.fillna(False)

        #lot
        p_lot = r40.str.contains('lot', regex=False)
        p_lot = p_lot.fillna(False)

        #valet
        p_valet = r40.str.contains('valet', regex=False)
        p_valet = p_valet.fillna(False)



        # EXPAND DIETARY RESTRICTIONS
        # setup
        r50 = pd.Series(data['dietaryRestrictions'])
        r50 = r50.fillna("None")

        #halal
        diet_halal = r50.str.contains('halal', regex=False)
        diet_halal = diet_halal.fillna("None")

        #kosher
        diet_kosher = r50.str.contains('kosher', regex=False)
        diet_kosher = diet_kosher.fillna("None")

        #soy-free
        diet_soy = r50.str.contains('soy-free', regex=False)
        diet_soy = diet_soy.fillna(False)

        #vegetarian
        diet_vegetarian = r50.str.contains('vegetarian', regex=False)
        diet_vegetarian = diet_vegetarian.fillna(False)

        #vegan
        diet_vegan = r50.str.contains('vegan', regex=False)
        diet_vegan = diet_vegan.fillna(False)

        #gluten-free
        diet_gluten = r50.str.contains('gluten-free', regex=False)
        diet_gluten = diet_gluten.fillna(False)

        #dairy-free
        diet_dairy = r50.str.contains('dairy-free', regex=False)
        diet_dairy = diet_dairy.fillna(False)



        #RECOMMENDED FOR
        #setup
        r60 = pd.Series(data['recommendedFor'])
        r60 = r60.fillna("None")

        #dessert
        rec_dessert = r60.str.contains('dessert', regex=False)
        rec_dessert = rec_dessert.fillna(False)

        #lateNight
        rec_latenight = r60.str.contains('latenight', regex=False)
        rec_latenight = rec_latenight.fillna(False)

        #lunch
        rec_lunch = r60.str.contains('lunch', regex=False)
        rec_lunch = rec_lunch.fillna(False)

        #dinner
        rec_dinner = r60.str.contains('dinner', regex=False)
        rec_dinner = rec_dinner.fillna(False)

        #brunch
        rec_brunch = r60.str.contains('brunch', regex=False)
        rec_brunch = rec_brunch.fillna(False)

        #breakfast
        rec_breakfast = r60.str.contains('breakfast', regex=False)
        rec_breakfast = rec_breakfast.fillna(False)



        #MERGE THE COLUMNS NOW:
        #print(data['dietaryRestrictions'])
        removed = data.drop(columns=['recommendedFor', 'dietaryRestrictions', 'parking', 'ambience'])
        removed1 = removed.replace("[]", "None")
        removed2 = removed1.fillna("None")

        #merge this bullshit
        removed2['amb_romantic'] = amb_romantic.values
        removed2['amb_intimate'] = amb_intimate.values
        removed2['amb_touristy'] = amb_touristy.values
        removed2['amb_trendy'] = amb_trendy
        removed2['amb_classy'] = amb_classy
        removed2['amb_casual'] = amb_casual
        removed2['amb_divey'] = amb_divey
        removed2['amb_hipster'] = amb_hipster
        removed2['amb_upscale'] = amb_upscale

        removed2['p_garage'] = p_garage
        removed2['p_valet'] = p_valet
        removed2['p_validate'] = p_validate
        removed2['p_street'] = p_street
        removed2['p_lot'] = p_lot

        removed2['diet_halal'] = diet_halal
        removed2['diet_gluten'] = diet_gluten
        removed2['diet_dairy'] = diet_dairy
        removed2['diet_kosher'] = diet_kosher
        removed2['diet_soy'] = diet_soy
        removed2['diet_vegetarian'] = diet_vegetarian
        removed2['diet_vegan'] = diet_vegan

        removed2['rec_breakfast'] = rec_breakfast
        removed2['rec_brunch'] = rec_brunch
        removed2['rec_dinner'] = rec_dinner
        removed2['rec_lunch'] = rec_lunch
        removed2['rec_latenight'] = rec_latenight
        removed2['rec_dessert'] = rec_dessert

        return removed2


if __name__ == '__main__':
        # CREATE AN OBJECT REFERENCE
        nb = NaiveBaes()

        # IMPORT THE DATASET
        # PASS IN THE MAIN AND THEN TAKE SUBSETS IF YOUR FEELING QUITE SAUCY
        df_main_test = sys.argv[3]
        df = nb.imp(df_main_test)

        # PASS IN TWO SEPERATES
        df_train_set = nb.imp(sys.argv[3])
        df_test_set = nb.imp(sys.argv[4])

        # print(df_train_set)
        # print(df_test_set)

        # SUBSET AND TEST
        # FOR THE SAUCY METHOD
        # train_model = df.iloc[:24722]
        # test_model = df.iloc[24788:24812]

        # GRAB THE ACTUAL OUTDOOR_SEATING COLUMN AND REMOVE SO WE CAN GUESS
        test_model = nb.preprocess(df_test_set)
        test_actual = test_model['outdoorSeating']
        test_model = test_model.drop(columns='outdoorSeating')

        # ATTEMPT TO PREPROCESS
        train_model = nb.preprocess(df_train_set)

        # print df
        mle = nb.buildMLE(train_model)

        for key, value in mle.items():
            print key
            print value
            print
            print
            print

        # calculate nbc omg fauoewjeiwa;fioeawjfioe;awjfiao;wejfoaiwf
        tested = nb.calcNBC(mle, test_model)
        print tested

        # score according to the principles of zero loss
        estimates = nb.zeroAndSquares(tested, test_actual)

        print "ZERO-ONE LOSS=%f" % estimates[1]
        print "SQUARED LOSS =%f" % estimates[0]


