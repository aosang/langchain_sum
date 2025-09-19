import 'dotenv/config'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { ChatDeepSeek } from '@langchain/deepseek'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { Send } from '@langchain/langgraph'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { Document } from '@langchain/core/documents'
import { collapseDocs, splitListOfDocs } from 'langchain/chains/combine_documents/reduce'

// 加载本地文档，模拟从数据库获取文档内容
async function loadMarkdownWithLoader(filePath) {
  const loader = new TextLoader(filePath)
  return await loader.load()
}

const doc = await loadMarkdownWithLoader('./data/blog.md')
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200
})
const splitDocs  = await textSplitter.splitDocuments(doc)

// 检查API密钥
if (!process.env.DEEPSEEK_API_KEY) {
  console.error('❌ 错误: 未找到 DEEPSEEK_API_KEY 环境变量')
  console.log('请创建 .env 文件并添加: DEEPSEEK_API_KEY=your_api_key_here')
  process.exit(1)
}

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
  temperature: 0,
  apiKey: process.env.DEEPSEEK_API_KEY
})

const mapSummaries = (state) => {
  // state.contents 是分割出来的文档内容数组
  return state.contents.map(
    (content) => new Send('generateSummary', { content }) // 每个文档内容都调用 generateSummary
  )
}

const mapPrompt = ChatPromptTemplate.fromMessages([
  ['user', '请对以下内容进行总结，只输出总结内容，不要包含任何说明、分析或建议：\n\n{context}'],
])

// 根据文档，生成总结
const generateSummary = async (state) => {
  // state.content 是分割出来的一段文档内容
  const prompt = await mapPrompt.invoke({ context: state.content })
  const response = await llm.invoke(prompt)
  
  const summary = String(response.content)
  return { summaries: [summary] }
}

const collectSummaries = async (state) => {
  return {
    collapsedSummaries: state.summaries.map(
      (summary) => new Document({ pageContent: summary })
    ),
  }
}

let tokenMax = 1500 // 设置最大 token 限制
async function lengthFunction(documents) {
  const tokenCounts = await Promise.all(
    documents.map(async (doc) => {
      return doc.pageContent.length
    })
  )
  return tokenCounts.reduce((sum, count) => sum + count, 0)
}

// 继续合并，还是生成最终总结？
async function shouldCollapse(state) {
  let numTokens = await lengthFunction(state.collapsedSummaries)
  if (numTokens > tokenMax) {
    return 'collapseSummaries'
  } else {
    return 'generateFinalSummary'
  }
}

const reducePrompt = ChatPromptTemplate.fromMessages([
  [
    'user',
    `下面是一组总结:
    {docs}
    
    请将这些总结合并成一个完整的总结。要求：
    1. 只输出最终总结内容
    2. 不要包含任何元信息、说明或分析
    3. 不要提及"改写"、"优化"等过程信息
    4. 直接给出总结结果`,
  ],
])

async function _reduce(input) {
  const prompt = await reducePrompt.invoke({ docs: input })
  const response = await llm.invoke(prompt)
  return String(response.content)
}

// 生成最后的总结
const generateFinalSummary = async (state) => {
  const response = await _reduce(state.collapsedSummaries)
  return { finalSummary: response }
}

const collapseSummaries = async (state) => {
  const docLists = splitListOfDocs(
    state.collapsedSummaries,
    lengthFunction,
    tokenMax
  )
  const results = []
  
  for (let i = 0; i < docLists.length; i++) {
    const docList = docLists[i]
    const collapsed = await collapseDocs(docList, _reduce) // 把 docList 中的文档合并为一个文档
    results.push(collapsed)
  }

  return { collapsedSummaries: results }
}


import { StateGraph, Annotation } from '@langchain/langgraph'

const OverallState = Annotation.Root({
  contents: Annotation,
  summaries: Annotation({
    reducer: (state, update) => state.concat(update),
  }),
  collapsedSummaries: Annotation,
  finalSummary: Annotation,
})

const graph = new StateGraph(OverallState)
  .addNode('generateSummary', generateSummary)
  .addNode('collectSummaries', collectSummaries)
  .addNode('collapseSummaries', collapseSummaries)
  .addNode('generateFinalSummary', generateFinalSummary)
  .addConditionalEdges('__start__', mapSummaries, ['generateSummary'])
  .addEdge('generateSummary', 'collectSummaries')
  .addConditionalEdges('collectSummaries', shouldCollapse, [
    'collapseSummaries',
    'generateFinalSummary',
  ])
  .addConditionalEdges('collapseSummaries', shouldCollapse, [
    'collapseSummaries',
    'generateFinalSummary',
  ])
  .addEdge('generateFinalSummary', '__end__')
  
const app = graph.compile()

let finalSummary = null
let summaryCount = 0
let totalSteps = splitDocs.length + 3 // 文档片段数 + 收集 + 合并 + 最终总结
let startMessageShown = false

// 创建动态加载条
function showLoadingBar(current, total, message) {
  const percentage = Math.round((current / total) * 100)
  const barLength = 30
  const filledLength = Math.round((current / total) * barLength)
  const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength)
  
  // 创建旋转动画
  const spinners = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
  const spinner = spinners[current % spinners.length]
  
  process.stdout.write(`\r${spinner} ${message} [${bar}] ${percentage}%`)
}


let currentStep = 0
let waitingInterval = null

// 显示开始消息
console.log('🚀 开始文档总结流程...\n')

for await (const step of await app.stream(
  { contents: splitDocs.map((doc) => doc.pageContent) },
  { recursionLimit: 10 }
)) {
  // 清除之前的等待动画
  if (waitingInterval) {
    clearInterval(waitingInterval)
    waitingInterval = null
  }
  
  // 如果是生成总结步骤
  if (step.hasOwnProperty('generateSummary')) {
    summaryCount++
    currentStep++
    showLoadingBar(currentStep, totalSteps, '⚡ 正在分析文档片段')
  }
  
  // 如果是收集总结步骤
  if (step.hasOwnProperty('collectSummaries')) {
    currentStep++
    // 如果是第一次进入收集阶段，清除开始消息
    if (!startMessageShown) {
      // 清除控制台，移除开始消息
      console.clear()
      startMessageShown = true
    }
    showLoadingBar(currentStep, totalSteps, '📋 正在收集总结内容')
  }
  
  // 如果是合并总结步骤
  if (step.hasOwnProperty('collapseSummaries')) {
    currentStep++
    showLoadingBar(currentStep, totalSteps, '🔄 正在合并总结内容')
  }
  
  // 如果是最终总结步骤
  if (step.hasOwnProperty('generateFinalSummary')) {
    currentStep++
    showLoadingBar(currentStep, totalSteps, '✨ 正在生成最终总结')
    finalSummary = step.generateFinalSummary
  }
  
  // 添加短暂延迟
  await new Promise(resolve => setTimeout(resolve, 200))
}

// 清除加载条
process.stdout.write('\r' + ' '.repeat(100) + '\r')

// 流式显示最终总结（打字效果）
if (finalSummary) {
  // 处理不同类型的 finalSummary
  let summaryText = ''
  if (typeof finalSummary === 'string') {
    summaryText = finalSummary
  } else if (typeof finalSummary === 'object' && finalSummary !== null) {
    // 如果是对象，尝试提取内容
    if (finalSummary.finalSummary) {
      summaryText = String(finalSummary.finalSummary)
    } else if (finalSummary.content) {
      summaryText = String(finalSummary.content)
    } else {
      summaryText = JSON.stringify(finalSummary, null, 2)
    }
  } else {
    summaryText = String(finalSummary)
  }
  
  // 打字效果显示
  // console.log('📄 最终总结:')
  
  const words = summaryText.split('')
  for (let i = 0; i < words.length; i++) {
    process.stdout.write(words[i])
    await new Promise(resolve => setTimeout(resolve, 30)) // 逐字符显示效果
  }
  console.log('\n')
  console.log('─'.repeat(50))
} else {
  console.log('❌ 未能生成最终总结')
}
