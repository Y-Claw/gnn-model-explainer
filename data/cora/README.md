This is a dirty graph from http://www.cs.utexas.edu/users/ml/riddle/data.html

Cora is a segmented citation dataset based on the Cora research paper search engine. Provided by William Cohen.

There are 2319 vertices(cora.dv) and 6445 edges(cora.de), while 17184(cora.dd) groundTruth are provided, which means 17184 pairs of nodes are duplicate.

After eliminating duplication(run src/preprocess/EliminateDuplication.py), there remains 1136 vertices(cora.v), 2059 edges(cora.e).

.v file:
 label:
   0: paper
   1: author
   2: conference
   3: publisher
 attributes:
   paper:		name	year	pages
   author:		name
   conference:	name
   publisher:	name

.e file:
 label:
   0: paper-author          (type: 0-1)
   1: paper-conference      (type: 0-2)
   2: paper-publisher       (type: 0-3)