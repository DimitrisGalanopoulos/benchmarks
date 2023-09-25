#!/usr/bin/awk -f

BEGIN {
    num_lines = 0
    lines_m[0] = ""
    lines_spmv[0] = ""
    matrix_prev = ""

    buf_m_mae = -1
    buf_m_max_ae = -1
    buf_m_mse = -1
    buf_m_mape = -1
    buf_m_smape = -1

    buf_spmv_mae = -1
    buf_spmv_max_ae = -1
    buf_spmv_mse = -1
    buf_spmv_mape = -1
    buf_spmv_smape = -1
}

/^File:/ {
    matrix = $2
    sub(".*/", "", matrix)
    sub("\\..*$", "", matrix)
    if (matrix != matrix_prev)
    {
        matrix_prev = matrix
        num_lines++

        buf_m_mae = -1
        buf_m_max_ae = -1
        buf_m_mse = -1
        buf_m_mape = -1
        buf_m_smape = -1

        buf_spmv_mae = -1
        buf_spmv_max_ae = -1
        buf_spmv_mse = -1
        buf_spmv_mape = -1
        buf_spmv_smape = -1
    }
}


/^errors matrix/ {
    num_fields = split($0, tok, "=")

    mae = tok[2]
    max_ae = tok[3]
    mse = tok[4]
    mape = tok[5]
    smape = tok[6]
    # if (buf_m_smape < 0 || smape > buf_m_smape)
    if (buf_m_smape < 0 || smape < buf_m_smape)
    {
        buf_m_mae = mae
        buf_m_max_ae = max_ae
        buf_m_mse = mse
        buf_m_mape = mape == "inf" ? "+inf" : mape
        buf_m_smape = smape == "inf" ? "+inf" : smape
        lines_m[num_lines] = sprintf("%g\t%g\t%g\t%g\t%g", buf_m_mae, buf_m_max_ae, buf_m_mse, buf_m_mape, buf_m_smape)
    }
}

/^errors spmv/ {
    num_fields = split($0, tok, "=|,")

    mae = tok[2]
    max_ae = tok[4]
    mse = tok[6]
    mape = tok[8]
    smape = tok[10]
    # if (buf_spmv_smape < 0 || smape > buf_spmv_smape)
    if (buf_spmv_smape < 0 || smape < buf_spmv_smape)
    {
        buf_spmv_mae = mae
        buf_spmv_max_ae = max_ae
        buf_spmv_mse = mse
        buf_spmv_mape = mape == "inf" ? "+inf" : mape
        buf_spmv_smape = smape == "inf" ? "+inf" : smape
        lines_spmv[num_lines] = sprintf("%g\t%g\t%g\t%g\t%g", buf_spmv_mae, buf_spmv_max_ae, buf_spmv_mse, buf_spmv_mape, buf_spmv_smape)
    }
}

END {
    if (1 in lines_m)
    {
        for (i=1;i<=num_lines;i++)
            printf("%s\t%s\n", lines_m[i], lines_spmv[i])
    }
    else
        for (i=1;i<=num_lines;i++)
            printf("%s\n", lines_spmv[i])
}


