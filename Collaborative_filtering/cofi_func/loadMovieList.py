def loadMovieList():
    with open("../data/movie_ids.txt", 'r', encoding = 'Windows 1252') as f:
        data = f.read()

    temp_movielist = {}
    movielist = {}
    row = ''
    count = 1

    for i in data:
        if i != '\n':
            row += i
        else:
            temp_movielist[count] = row
            row = ''
            count += 1

    for index in temp_movielist:
        s = temp_movielist[index]
        for i in s:
            if i != ' ':
                s = s[1:]
            else:
                s = s[1:]
                movielist[index] = s
                break
    return movielist
