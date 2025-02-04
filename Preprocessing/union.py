def GPU_Union(name, subseqLen, fragmentNum, fragmentComp, fragmentNum, step):

    lib = ctypes.CDLL("C:\\Project\\Preprocessing\\CUDA_TS.dll")

    column_names = ["NNdistance_256_1","NNindex_256_1", "NNindexFragment_256_1", "fragment", "num"]

    cTs1 = (ctypes.c_double * (fragmentNum + subseqLen - 1))()
    cMu1 = (ctypes.c_double * fragmentNum)()
    cSigma1 = (ctypes.c_double * fragmentNum)()
    cMp1 = (ctypes.c_double * fragmentNum)()
    cMpIndex1 = (ctypes.c_int64 * fragmentNum)()

    cTs2 = (ctypes.c_double * (fragmentNum + subseqLen - 1))()
    cMu2 = (ctypes.c_double * fragmentNum)()
    cSigma2 = (ctypes.c_double * fragmentNum)()
    cMp2 = (ctypes.c_double * fragmentNum)()
    cMpIndex2 = (ctypes.c_int64 * fragmentNum)()

    cSubseqLen = ctypes.c_int64(subseqLen)

    for i in range(0, fragmentComp + 1, step):
        iStepOld = i
        iFragment = i + step +1
        if iFragment > fragmentNum + 1:
            iFragment = fragmentNum + 1

        ts1 = client.query_np(f'SELECT val_256_1 FROM default.TS_THERMALSENSOR WHERE fragment > {iStepOld} and fragment < {iFragment} ORDER BY fragment').transpose()
        muSigma1 = client.query_np(f'SELECT mu_256_1, sigma_256_1 FROM default.FT_THERMALSENSOR WHERE fragment > {iStepOld} AND fragment < {iFragment} ORDER BY fragment').transpose()
        mp1 = client.query_np(f'SELECT NNdistance_256_1, FROM default.MP_THERMALSENSOR WHERE fragment > {iStepOld} AND fragment < {iFragment} ORDER BY fragment').transpose()
        mpIndex1 = client.query_np(f'SELECT NNindex_256_1, NNindexFragment_256_1, fragment, num FROM default.MP_THERMALSENSOR WHERE fragment > {iStepOld} AND fragment < {iFragment} ORDER BY fragment').transpose()

        for j in range(fragmentComp-1, fragmentNum + 1, step):
            jStepOld = j
            jFragment = j + step +1
            if jFragment > fragmentNum + 1:
                jFragment = fragmentNum + 1
            
            ts2 = client.query_np(f'SELECT val_256_1 FROM default.TS_THERMALSENSOR WHERE fragment > {jStepOld} and fragment < {jFragment} ORDER BY fragment').transpose()
            muSigma2 = client.query_np(f'SELECT mu_256_1, sigma_256_1 FROM default.FT_THERMALSENSOR WHERE fragment > {jStepOld} AND fragment < {jFragment} ORDER BY fragment').transpose()
            mp2 = client.query_np(f'SELECT NNdistance_256_1, FROM default.MP_THERMALSENSOR WHERE fragment > {jStepOld} AND fragment < {jFragment} ORDER BY fragment').transpose()
            mpIndex2 = client.query_np(f'SELECT NNindex_256_1, NNindexFragment_256_1, fragment, num FROM default.MP_THERMALSENSOR WHERE fragment > {jStepOld} AND fragment < {jFragment} ORDER BY fragment').transpose()

            stepTs1 = subseqLen - 1
            for io in range(0, muSigma1.shape[1], fragmentNum):

                ctypes.memmove(cTs1, ts1[0][io + stepTs1 - (subseqLen - 1):io + stepTs1 + fragmentNum].ctypes.data, ts1[0][io + stepTs1 - (subseqLen - 1):io + stepTs1 + fragmentNum].nbytes)
                ctypes.memmove(cMu1, muSigma1[0][io:io + fragmentNum].ctypes.data, muSigma1[0][io:io + fragmentNum].nbytes)
                ctypes.memmove(cSigma1, muSigma1[1][io:io + fragmentNum].ctypes.data, muSigma1[1][io:io + fragmentNum].nbytes)
                ctypes.memmove(cMp1, mp1[0][io:io + fragmentNum].ctypes.data, mp1[0][io:io + fragmentNum].nbytes)
                ctypes.memmove(cMpIndex1, mpIndex1[0][io:io + fragmentNum].ctypes.data, mpIndex1[0][io:io + fragmentNum].nbytes)

                stepTs2 = subseqLen - 1

                for jo in range(0, muSigma2.shape[1], fragmentNum):

                    ctypes.memmove(cTs2, ts2[0][jo + stepTs2 - (subseqLen - 1):jo + stepTs2 + fragmentNum].ctypes.data, ts2[0][jo + stepTs2 - (subseqLen - 1):jo + stepTs2 + fragmentNum].nbytes)
                    ctypes.memmove(cMu2, muSigma2[0][jo:jo + fragmentNum].ctypes.data, muSigma2[0][jo:jo + fragmentNum].nbytes)
                    ctypes.memmove(cSigma2, muSigma2[1][jo:jo + fragmentNum].ctypes.data, muSigma2[1][jo:jo + fragmentNum].nbytes)
                    ctypes.memmove(cMp2, mp2[0][jo:jo + fragmentNum].ctypes.data, mp2[0][jo:jo + fragmentNum].nbytes)
                    ctypes.memmove(cMpIndex2, mpIndex2[0][jo:jo + fragmentNum].ctypes.data, mpIndex2[0][jo:jo + fragmentNum].nbytes)

                    lib.cudaRun_Union(cSubseqLen, cTs1, cMu1, cSigma1, cMp1, cMpIndex1, cTs2, cMu2, cSigma2, cMp2, cMpIndex2)

                    mp2[0][jo:jo + fragmentNum] = np.ctypeslib.as_array(cMp2)
                    mpIndex2[0][jo:jo + fragmentNum] = np.ctypeslib.as_array(cMpIndex2)

                    stepTs2 = stepTs2 + subseqLen -1

                mp1[0][jo:jo + fragmentNum] = np.ctypeslib.as_array(cMp1)
                mpIndex1[0][jo:jo + fragmentNum] = np.ctypeslib.as_array(cMpIndex1)

                stepTs1 = stepTs1 + subseqLen -1

            mp2 = mp2.transpose()
            mpIndex2 = mpIndex2.transpose()
            mp2 = np.hstack((mp2, mpIndex2))

            client.command(f"DELETE FROM MP_THERMALSENSOR WHERE fragment > {jStepOld} and fragment < {jFragment}")
            client.insert(f'default.MP_{name}', mp2, column_names=column_names)
        
        mp1 = mp1.transpose()
        mpIndex1 = mpIndex1.transpose()
        mp1 = np.hstack((mp1, mpIndex1))

        client.command(f"DELETE FROM MP_THERMALSENSOR WHERE fragment > {iStepOld} and fragment < {iFragment}")
        client.insert(f'default.MP_{name}', mp1, column_names=column_names)

