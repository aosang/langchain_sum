import 'dotenv/config'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { ChatDeepSeek } from '@langchain/deepseek'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { StringOutputParser } from '@langchain/core/output_parsers'
import { PromptTemplate } from '@langchain/core/prompts'

// 加载本地文档，模拟从数据库获取文档内容
async function loadMarkdownWithLoader(filePath) {
  const loader = new TextLoader(filePath)
  return await loader.load()
}

const doc = await loadMarkdownWithLoader('./data/article.md')
// console.log('doc content', doc[0].pageContent.substring(0, 300))

const llm = new ChatDeepSeek({
  model: 'deepseek-chat',
  temperature: 0,
  apiKey: process.env.DEEPSEEK_API_KEY
})

const prompt = PromptTemplate.fromTemplate(`
  简单总结这篇文章，200字以内
  <article>
  {context}
  </article>
`)

const chain = await createStuffDocumentsChain({
  llm: llm,
  prompt,
  outputParser: new StringOutputParser()
})

// const result = await chain.invoke({ context: doc })
// console.log('result', result)

const stream = await chain.stream({ context: doc })
for await (const chunk of stream) {
  process.stdout.write(chunk)
}