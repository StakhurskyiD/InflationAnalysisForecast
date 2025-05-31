import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_data(df: pd.DataFrame):
    """
    Будує кілька ОКРЕМИХ (не subplots) графіків за даними датафрейму.
    """

    # ------------ Нові КРОКИ ПЕРЕД ПОБУДОВОЮ ГРАФІКІВ ------------
    # Припустимо, що 'year' має бути int, 'inflation_index' - float
    # 1) Приведемо year до int (якщо вже не int)
    if df['year'].dtype != 'int64':
        df['year'] = (
            df['year']
            .astype(str)
            .str.replace(',', '.')     # На випадок 2021,0 -> 2021.0 -> "2021.0"
            .astype(float)
            .astype(int)
        )

    # 2) Приведемо inflation_index до float (заміна ком -> крапка)
    if df['inflation_index'].dtype != 'float':
        df['inflation_index'] = (
            df['inflation_index']
            .astype(str)
            .str.replace(',', '.')
            .astype(float)
        )

    # Переконаємося, що жоден рядок не має NaN
    df.dropna(subset=['year','inflation_index'], inplace=True)

    # 3) Будуємо
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(df['year'], df['inflation_index'], marker='o')
    plt.title("Динаміка Індексу інфляції")
    plt.xlabel("Рік")
    plt.ylabel("Індекс інфляції")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Якщо є 'ppi_mom_growth':
    if 'ppi_mom_growth' in df.columns:
        # Приведемо до float за аналогією
        if df['ppi_mom_growth'].dtype != 'float':
            df['ppi_mom_growth'] = (
                df['ppi_mom_growth']
                .astype(str)
                .str.replace(',', '.')
                .astype(float)
            )
        df.dropna(subset=['ppi_mom_growth'], inplace=True)

        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(df['year'], df['ppi_mom_growth'], marker='o')
        plt.title("Динаміка PPI (Month-over-Month Growth)")
        plt.xlabel("Рік")
        plt.ylabel("PPI MoM Growth, %")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Попередження: колонки 'ppi_mom_growth' не знайдено у DataFrame.")