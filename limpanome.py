#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def excluir_arquivo(caminho_completo):
    """
    Exclui um arquivo que contém underscore ou espaço no nome
    """
    nome_arquivo = os.path.basename(caminho_completo)
    
    try:
        os.remove(caminho_completo)
        print(f"✅ Excluído: '{nome_arquivo}'")
        return True
        
    except PermissionError:
        print(f"❌ ERRO: Sem permissão para excluir '{nome_arquivo}'")
        return False
    except Exception as e:
        print(f"❌ ERRO ao excluir '{nome_arquivo}': {e}")
        return False

def processar_pasta(caminho_pasta):
    """
    Processa recursivamente todos os arquivos da pasta
    """
    if not os.path.exists(caminho_pasta):
        print(f"❌ ERRO: A pasta '{caminho_pasta}' não existe!")
        return
    
    if not os.path.isdir(caminho_pasta):
        print(f"❌ ERRO: '{caminho_pasta}' não é uma pasta!")
        return
    
    print(f"\n🔍 Processando pasta: {caminho_pasta}")
    print("=" * 60)
    
    total_excluidos = 0
    
    # Lista para armazenar todos os arquivos a excluir
    arquivos_para_excluir = []
    
    # Percorre a árvore de diretórios
    for root, dirs, files in os.walk(caminho_pasta):
        # Adiciona arquivos com underscore ou espaço à lista
        for arquivo in files:
            if '_' in arquivo or ' ' in arquivo:
                caminho_completo = os.path.join(root, arquivo)
                arquivos_para_excluir.append(caminho_completo)
    
    # Processa arquivos
    if arquivos_para_excluir:
        print(f"\n📄 Encontrados {len(arquivos_para_excluir)} arquivo(s) com underscore (_) ou espaço ( ):")
        print("\nLista de arquivos que serão excluídos:")
        for arquivo in arquivos_para_excluir:
            nome = os.path.basename(arquivo)
            # Indica o motivo da exclusão
            marcadores = []
            if '_' in nome:
                marcadores.append('_')
            if ' ' in nome:
                marcadores.append('espaço')
            motivo = f"[{', '.join(marcadores)}]"
            print(f"  - {nome} {motivo}")
        
        # Confirmação adicional antes de excluir
        print(f"\n⚠️  ATENÇÃO: {len(arquivos_para_excluir)} arquivo(s) serão PERMANENTEMENTE excluídos!")
        print("\n🗑️  Excluindo arquivos...")
        for arquivo in arquivos_para_excluir:
            if excluir_arquivo(arquivo):
                total_excluidos += 1
    else:
        print("ℹ️  Nenhum arquivo com underscore (_) ou espaço ( ) foi encontrado.")
    
    # Resumo final
    print("\n" + "=" * 60)
    if total_excluidos > 0:
        print(f"✅ CONCLUÍDO: {total_excluidos} arquivo(s) excluído(s) com sucesso!")
    else:
        print("ℹ️  Nenhum arquivo foi excluído.")

def main():
    # Caminho da pasta a processar
    caminho = r'./img_base_captcha'
    
    print("🚀 SCRIPT DE EXCLUSÃO DE ARQUIVOS COM UNDERSCORE OU ESPAÇO")
    print("=" * 60)
    print(f"Pasta alvo: {caminho}")
    
    # Aviso importante
    print("\n" + "⚠️ " * 10)
    print("AVISO IMPORTANTE: Este script irá EXCLUIR PERMANENTEMENTE")
    print("todos os arquivos que contêm underscore (_) OU espaço ( ) no nome!")
    print("Esta ação NÃO PODE ser desfeita!")
    print("⚠️ " * 10)
    
    # Confirmação do usuário
    resposta = input("\nDeseja continuar? (s/n): ")
    
    if resposta.lower() in ['s', 'sim', 'yes', 'y']:
        processar_pasta(caminho)
    else:
        print("❌ Operação cancelada pelo usuário.")
        sys.exit(0)
    
    print("\n🏁 Script finalizado!")
    input("\nPressione ENTER para sair...")

if __name__ == "__main__":
    main()