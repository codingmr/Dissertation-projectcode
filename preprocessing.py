from sklearn import preprocessing

inputFile = "Alteromonas stellipolaris2.fna"
outputFile = "Amino.fna"

# read each line and put it into a single line

with open(inputFile, "r+") as f:
	strData = f.read()
	strData = strData.replace("\n", "")

	strData = [strData[i:i+3] for i in range(0, len(strData), 3)]
	strData = " ".join(strData)

	strData = strData.replace("TTT", "Phe")
	strData = strData.replace("TTC", "Phe")
	strData = strData.replace("TTA", "Leu")
	strData = strData.replace("TTG", "Leu")

	strData = strData.replace("CTT", "Leu")
	strData = strData.replace("CTC", "Leu")
	strData = strData.replace("CTA", "Leu")
	strData = strData.replace("CTG", "Leu")

	strData = strData.replace("ATT", "Ile")
	strData = strData.replace("ATC", "Ile")
	strData = strData.replace("ATA", "Ile")
	strData = strData.replace("ATG", "Met")
	
	strData = strData.replace("GTT", "Val")
	strData = strData.replace("GTC", "Val")
	strData = strData.replace("GTA", "Val")
	strData = strData.replace("GTG", "Val")

	strData = strData.replace("TCT", "Ser")
	strData = strData.replace("TCC", "Ser")
	strData = strData.replace("TCA", "Ser")
	strData = strData.replace("TCG", "Ser")

	strData = strData.replace("CCT", "Pro")
	strData = strData.replace("CCC", "Pro")
	strData = strData.replace("CCA", "Pro")
	strData = strData.replace("CCG", "Pro")

	strData = strData.replace("ACT", "Thr")
	strData = strData.replace("ACC", "Thr")
	strData = strData.replace("ACA", "Thr")
	strData = strData.replace("ACG", "Thr")

	strData = strData.replace("GCT", "Ala")
	strData = strData.replace("GCC", "Ala")
	strData = strData.replace("GCA", "Ala")
	strData = strData.replace("GCG", "Ala")

	strData = strData.replace("TAT", "Tyr")
	strData = strData.replace("TAC", "Tyr")
	strData = strData.replace("TAA", "Stp")
	strData = strData.replace("TAG", "Stp")
	
	strData = strData.replace("CAT", "His")
	strData = strData.replace("CAC", "His")
	strData = strData.replace("CAA", "Gin")
	strData = strData.replace("CAG", "Gin")

	strData = strData.replace("AAT", "Asn")
	strData = strData.replace("AAC", "Asn")
	strData = strData.replace("AAA", "Lys")
	strData = strData.replace("AAG", "Lys")

	strData = strData.replace("GAT", "Asp")
	strData = strData.replace("GAC", "Asp")
	strData = strData.replace("GAA", "Glu")
	strData = strData.replace("GAG", "Glu")

	strData = strData.replace("TGT", "Cys")
	strData = strData.replace("TGC", "Cys")
	strData = strData.replace("TGA", "Stp")
	strData = strData.replace("TGG", "Trp")

	strData = strData.replace("CGT", "Arg")
	strData = strData.replace("CGC", "Arg")
	strData = strData.replace("CGA", "Arg")
	strData = strData.replace("CGG", "Arg")

	strData = strData.replace("AGT", "Ser")
	strData = strData.replace("AGC", "Ser")
	strData = strData.replace("AGA", "Arg")
	strData = strData.replace("AGG", "Arg")

	strData = strData.replace("GGT", "Gly")
	strData = strData.replace("GGC", "Gly")
	strData = strData.replace("GGA", "Gly")
	strData = strData.replace("GGG", "Gly")

	strNew = strData

    	max_width = 50
    	result = ""
    	col = 0

        for word in strNew.split():
            end_col = col + len(word)
            if col != 0:
                end_col += 1
            if end_col > max_width: 
                result += '\n'
                col = 0    
            if col != 0:
                result += ' ' 
                col += 1
            result += word 
            col += len(word)
	strNew = result
	

with open(outputFile, "w") as f:
	f.write(strNew)

with open("Amino.fna") as f:
	strData = f.read()
	a1count = strData.count("Phe")
	a2count = strData.count("Leu")
	a3count = strData.count("Ile")
	a4count = strData.count("Met")
	a5count = strData.count("Val")
	a6count = strData.count("Ser")
	a7count = strData.count("Pro")
	a8count = strData.count("Thr")
	a9count = strData.count("Ala")	
	a10count = strData.count("Tyr")
	a11count = strData.count("Stp")
	a12count = strData.count("His")
	a13count = strData.count("Gin")
	a14count = strData.count("Asn")
	a15count = strData.count("Asp")
	a16count = strData.count("Glu")
	a17count = strData.count("Arg")
	a18count = strData.count("Gly")
	a19count = strData.count("Cys")
	a20count = strData.count("Hyp")
	a21count = strData.count("Trp")

	print "Phe count: ", a1count
	print "Leu count: ", a2count
	print "Ile count: ", a3count
	print "Trp count: ", a4count
	print "Hyp count: ", a5count
	print "Cys count: ", a6count
	print "Gly count: ", a7count
	print "Arg count: ", a8count
	print "Glu count: ", a9count
	print "Asp count: ", a10count
	print "Asn count: ", a11count
	print "Gin count: ", a12count
	print "His count: ", a13count
	print "Stp count: ", a14count
	print "Tyr count: ", a15count
	print "Ala count: ", a16count
	print "Thr count: ", a17count
	print "Pro count: ", a18count
	print "Ser count: ", a19count
	print "Val count: ", a20count
	print "Met count: ", a21count

	print "Total characters: ", len(strData.split())
	
