import os
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv


load_dotenv()
pool = None


def init_pool():
    """
    Initializes a global PostgreSQL connection pool using environment variables.

    This method configures a connection pool with parameters for host, port, 
    database name, and credentials. It establishes a pool size range of 1 to 5 
    connections and sets a connection timeout of 10 seconds. Upon initialization, 
    it verifies connectivity by executing a 'SELECT 1' query.

    Raises:
        RuntimeError: If the connection pool cannot be initialized or if 
                      the initial connectivity test fails.

    Global:
        pool (ConnectionPool): The initialized connection pool instance.
    """
    # get data from .env
    db_user = os.getenv("DB_USER").strip()
    db_password = os.getenv("DB_PASSWORD").strip()
    db_name = os.getenv("DB_NAME").strip()
    db_host = os.getenv("DB_HOST").strip()
    db_port = os.getenv("DB_PORT").strip()

    conninfo = (
        f"host={db_host} "
        f"port={db_port} "
        f"dbname={db_name} "
        f"user={db_user} "
        f"password={db_password} "
        f"connect_timeout=10"   # max time for client to wait to connect with db
    )
    try:
        global pool
        pool = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=5,
            timeout=30,     # wait max for 30 sec if all conns are not avialable 
            open=True
        )

        pool.wait()     # wait untill min num of conns will be created 

        # test whether the pool was actually created 
        with pool.connection() as conn:
            conn.execute("SELECT 1")
        
        print("Successfully created a pool!")

    except Exception as e:
        raise RuntimeError(f"Error with connecting to DB: {e}")
