SELECT a.date, a.permno, b.ticker, a.prc, a.cfacpr, a.ret, a.retx
                     FROM crsp.dsf a JOIN crsp.stocknames b ON a.permno=b.permno
                     WHERE b.ticker IN %(tickers)s
                     AND a.date BETWEEN %(start)s AND %(end)s
                     AND a.date BETWEEN b.namedt AND b.nameenddt
                     ORDER BY a.permno, a.date