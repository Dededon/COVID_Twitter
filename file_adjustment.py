import os
import pandas as pd


def adjust_retweet_files():
    retweet_files = os.listdir('data/tweets/retweets/')
    for file in retweet_files:
        if 'adjusted' in file:
            continue
        if '.csv' not in file:
            continue
        print (file)
        with open ('data/tweets/retweets/adjusted_'+file,'w') as writer:

            with open('data/tweets/retweets/'+file,'r') as f:
                rows = 0


                for row in f:
                    row_val = ['']*19

                    rows+=1

                    cells = row.strip().split(',')

                    if len (cells) == 19:
                        for v in cells:
                            writer.write(v+',')
                        writer.write('\n')
                    else:
                        val_cnt = 18

                        start = False


                        ext_txt = ''
                        for i in range (len(cells),0,-1):


                            if not start:
                                combine = cells[i-1]


                            just_stopped = False

                            if ']' in cells[i-1] and '[' not in cells[i-1]:
                                start = True

                            if '[' in cells[i-1] and ']' not in cells[i-1]:
                                start = False

                                just_stopped = True

                            if start:
                                combine = cells[i-1]+combine

                            if just_stopped:

                                combine = cells[i-1]+combine

                                if val_cnt>=0:
                                    row_val[val_cnt] = combine 

                                val_cnt -= 1

                                just_stopped = True

                                continue

                            if not start:

                                if val_cnt>=0:
                                    row_val[val_cnt] = combine

                                val_cnt -= 1

                            if val_cnt< 0:
                                ext_txt = cells[i-1] + ext_txt
            #             print('ext_txt' , ext_txt)
            #             print ('row', row_val)
                        row_val[0] = ext_txt #+ row_val[0]

                        if len(row_val) > 19:
                            print('p[s]')


                        for v in row_val:
                            writer.write(v+',')
                        writer.write('\n')

                print ('rows', rows)


def adjust_tweet_files():
    tweet_files =[x for x in  os.listdir('data/tweets/') if '.csv' in x]

    for file in tweet_files:
        if 'adjusted' in file:
            continue
        if '.csv' not in file:
            continue
        print (file)    

        rows = 0    

        with open('data/tweets/adjusted_' +file , 'w') as writer:

            print ('writing')

            with open('data/tweets/'+file,'r') as f:
                rows = 0


                for row in f:
                    row_val = ['']*19

                    rows+=1
                    cells = row.rstrip().split(',')

                    final_row = ['']*16


                    if len(cells)>16:
        #                 print (cells)


                        isqoute = str(cells[2])

        #                 print ('\n\b')
                        for index in range(0,4):
            #                 print (cells[index])
                            final_row[index] = cells[index]

                        backwards = 15 

                        for index in range(len(cells), len(cells)- 10,-1):
            #                 print (index, cells[index-1], backwards, len(final_row))
                            final_row[backwards] = cells[index-1]    
                            backwards -=1

                        start = True
                        text = cells[4]
                        qt = ''

                        text_end = -1
                        continue_merge = True

                        for index in range(5,len(cells)- 10):
            #                 print ('need to adjust', index)
                            curr = cells[index]

                            if '"' in curr and continue_merge:
                                text = text +  ' ' +cells[index]

                                final_row[4] = text.replace('"', '')
                                continue_merge = False
                                text_end = index+1
                                break

                            else:
                                text = text +  ' ' +cells[index]

            #                 print (curr)


                        for index in range(text_end,len(cells)- 10):

                                qt = qt +  ' ' +cells[index]
                                final_row[5] = qt.replace('"', '')


                        if isqoute == 'False':
                            text = ''
                            for index in range(4,len(cells)- 11):
                                text = text + ' ' + cells[index] 
                            final_row[4] = text.replace('"', '')
                            final_row[5] = '""'

            #             print ('merged -->', text )
            #             print("----")
            #             print ('qoute text merged -->', qt )

        #                 print ('\n\n')
        #                 print ('row: --- ', row)
        #                 print (final_row, len(final_row))
        #                 print ('\n\n')


        #                 print ("++++++++++")


                    else:
                        final_row = cells



                    for v in final_row:
                        writer.write(str(v)+',')
                    writer.write('\n')        


        #             if rows==20:
        #                 break
            print ('done')

