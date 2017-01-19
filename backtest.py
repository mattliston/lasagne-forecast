import csv
import numpy as np
import argparse
#load open/close prices + date&symbol
#load signal + date&symbol
#compare signal: if active start tracking 
#track percent gain
#use tuple of sym,date to map to signal

#parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--buy', help='strength of signal to long', default=15, type=int)
parser.add_argument('--short', help='strength of signal to short', default=5, type=int)
parser.add_argument('--sell_long', help='strength of signal to exit long position', default=8, type=int)
parser.add_argument('--sell_short', help='strength of signal to exit short position', default=12, type=int)
args = parser.parse_args()

performance = 0
signalmap = {}
with open('nansignals.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        identifier = tuple([row[0],row[1]])
        if identifier not in signalmap:
            signalmap[identifier] = row[2]
        else:
            continue
#print signalmap #maps all signalled dates

previous_close = []
previous_ticker = 'A' #to determine end of stock history
open_position = 0
active = False
long_position = False
short_position = False
with open('WIKI_20161229.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
#add logic if row[0] doesnt equal row of previous
        if row[0]!=previous_ticker:
            if active and long_position:
                try:
#                    print 'ticker', row[0], 'date', row[1]
                    change = ((float(previous_close.pop())/float(open_position))-1)*100
                    print 'change', change
                    performance += change
                    print 'performance', performance
                    active=False
                    long_position=False #position closed out
                except ValueError:
                    for i in range(0,len(previous_close)):
                        last = previous_close.pop()
                        if last:
                           break                                     
                    print last, open_position
                    change = ((float(last)/float(open_position))-1)*100
                    print 'change', change
                    performance += change
                    print 'performance', performance
                    active=False
                    long_position=False #position closed out
            elif active and short_position:
                change = ((float(open_position)/float(previous_close.pop()))-1)*100
                print 'change', change
                performance += change
                print 'performance', performance
                active=False
                short_position=False
            else:
                pass
        else:
            pass

        current = tuple([row[0],row[1]])
        try:
            signal = signalmap[current]
            if not active:
                if signal == -1:
                    previous_ticker = row[0]
                    continue
                elif signal >= args.buy:
                    try:
                        open_position = float(row[2])
                        active = True
                        long_position = True
                    except ValueError:
                        previous_ticker = row[0]
                        continue
                elif signal <= args.short:
                    try:
                        open_position = float(row[2])
                        active = True
                        short_position = True
                    except ValueError:
                        previous_ticker = row[0]
                        continue
                else:
                    pass
            else: #active
                if signal == -1:
                    previous_ticker = row[0]
                    continue
                elif signal <= args.sell_long:
                    change = ((float(open_position)/float(row[2]))-1)*100
                    print 'change', change
                    performance += change
                    print 'performance', performance
                    active = False
                    long_position = False
                elif signal >= args.sell_short:
#                    print 'row[2]', row[2], 'open_position', open_position
#                    print 'symbol', row[0], 'date', row[1]
                    try:
                        change = ((float(row[2])/float(open_position))-1)*100
                        print 'change', change
                        performance += change
                        print 'performance', performance
                        active = False
                        short_position = False
                    except ZeroDivisionError: #it shorted a stock that went bankrupt
                        open_position = 0.01
                        change = ((float(row[2])/float(open_position))-1)*100
                        print 'change', change
                        performance += change
                        print 'performance', performance
                        active = False
                        short_position = False
                        
                    except ValueError:
                        pass
                else:
                    pass
        
        except KeyError:
            continue
        previous_ticker = row[0]
        previous_close.append(row[5])       
    #for each signalled date, retrieve signal buy at open or sell at close
 
