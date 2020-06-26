.v file:
 label:
   0: movie
   1: director
   2: writer
   3: actor
 attributes:
   movie:		title  year
   director:   name
   writer:     name
   actor:      name

.e file:
 label:
   0: movie-director    (type: 0-1)
   1: movie-writer      (type: 0-2)
   2: movie-actor       (type: 0-3)