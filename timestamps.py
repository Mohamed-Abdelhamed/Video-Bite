def generateTimestamps(timestamp):
    maxlength = max(timestamp, key=lambda ev: ev['end'])
    
    #filter timestamps that's taking more than 80%.
    if len(timestamp) > 2: 
        filtered_timestamp = [x for x in timestamp if ( (x['end']-x['start'])/maxlength['end'] )*100 < 80]
    
    #sort ascending 
    timesorted = sorted(filtered_timestamp, key=lambda k: k['start'])
    
    #call mergeintervals for removing overlaps
    merged = mergeIntervals(timesorted)
    
    #get timestamp with largest weight in given interval.
    filtered_timestamps = []
    for x in merged:
        maxweight = max(x[1], key=lambda ev: ev['weight'])
        filtered_timestamps.append({'start':x[0][0] , 'end': x[0][1] ,'sentence' : timesorted[maxweight['index']]['sentence'] })
    
    return filtered_timestamps


def mergeIntervals(arr):
        # array to hold the merged intervals
        time=[]
        times=[]
        m = []
        s = -10000
        max = -100000                                      
        for i in range(len(arr)):
            a = arr[i]
            a['weight'] = a['end'] - a['start']
            if a['start'] > max:
                if i != 0:
                    times.append([[s,max],time])
                    time=[]
                    m.append([s,max])
                max = a['end']
                s = a['start']
            else:
                if a['end'] >= max:
                    max = a['end']
            time.append({'index':i,'weight':a['weight']})
         
        if max != -100000 and [s, max] not in m:
            times.append([[s,max],time])
            time=[]
            m.append([s, max])    
        return times


